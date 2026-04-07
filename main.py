import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time
import os

# --- HARDWARE SETUP ---
GPIO.setmode(GPIO.BCM)
ENA, IN1, IN2 = 12, 23, 24
ENB, IN3, IN4 = 13, 17, 27
GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)
pwmA = GPIO.PWM(ENA, 1000); pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0); pwmB.start(0)

# --- CONFIGURATION ---
SAVE_DIR = "templates"
ROI_SPLIT = 0.65
N_SCAN = 5          # Reduced for faster checking
MATCH_THRESH = 0.75 # Slightly lowered to account for camera noise
DEADZONE = 5
SMOOTHING = 0.25 

# --- GLOBALS ---
last_error = 0
steer_filter = 0
frame_count = 0
stop_until = 0
templates = {}

def load_templates():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR, exist_ok=True)
    tpls = {}
    files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(".png")]
    print(f"✅ Loaded {len(files)} templates.")
    for f in files:
        img = cv2.imread(os.path.join(SAVE_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is not None: tpls[f] = img
    return tpls

templates = load_templates()

def set_motors(l, r):
    l, r = np.clip(l, -100, 100), np.clip(r, -100, 100)
    GPIO.output(IN1, GPIO.LOW if l >= 0 else GPIO.HIGH)
    GPIO.output(IN2, GPIO.HIGH if l >= 0 else GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW if r >= 0 else GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH if r >= 0 else GPIO.LOW)
    pwmA.ChangeDutyCycle(abs(l))
    pwmB.ChangeDutyCycle(abs(r))

def preprocess_template(crop_gray, size=(120, 120)):
    crop_eq = cv2.equalizeHist(crop_gray)
    h, w = crop_eq.shape
    diff = abs(h - w)
    t, b, l, r = (diff//2, diff-diff//2, 0, 0) if h < w else (0, 0, diff//2, diff-diff//2)
    crop_sq = cv2.copyMakeBorder(crop_eq, t, b, l, r, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255), size)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (480, 360)})
picam2.configure(config)
picam2.start()

print("Systems Online. Press ENTER to start!")
input(">>> ")

try:
    cur_sym = "None"
    
    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape
        disp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        current_time = time.time()

        # 1. ─── SYMBOL DETECTION ──────────────────
        # Always process the debug view, but only match when not paused
        roi_y_top = int(h * ROI_SPLIT)
        sym_roi = gray[0:roi_y_top, :]
        sym_blur = cv2.GaussianBlur(sym_roi, (7, 7), 0)
        sym_bin = cv2.adaptiveThreshold(sym_blur, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 81, 10)
        
        cv2.imshow("Shape Debug (Binary)", sym_bin)

        if current_time > stop_until and frame_count % N_SCAN == 0:
            contours, _ = cv2.findContours(sym_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 1500 < area < 30000: 
                    x, y, wb, hb = cv2.boundingRect(c)
                    aspect_ratio = wb / float(hb)
                    
                    if 0.5 < aspect_ratio < 1.8 and templates:
                        proc_crop = preprocess_template(sym_roi[y:y+hb, x:x+wb])
                        best_score, best_name = 0, "None"
                        for name, tpl in templates.items():
                            res = cv2.matchTemplate(proc_crop, tpl, cv2.TM_CCOEFF_NORMED)
                            _, score, _, _ = cv2.minMaxLoc(res)
                            if score > best_score:
                                best_score, best_name = score, name.split('.')[0]
                        
                        if best_score > MATCH_THRESH:
                            cur_sym = best_name
                            set_motors(0, 0)
                            stop_until = current_time + 5
                            
                            # UI Popup
                            result_win = np.zeros((200, 450, 3), dtype=np.uint8)
                            cv2.putText(result_win, f"SHAPE: {cur_sym}", (20, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(result_win, f"CONF: {int(best_score*100)}%", (20, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                            cv2.imshow("Detection Result", result_win)
                            break

        # 2. ─── MOTOR LOGIC ────────────────────────
        line_roi_y = int(h * ROI_SPLIT)
        line_roi = gray[line_roi_y : line_roi_y + 50, :]
        blur = cv2.GaussianBlur(line_roi, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if ret > 150: thresh = np.zeros_like(thresh)
        cv2.imshow("Line Debug (Threshold)", thresh)

        if current_time < stop_until:
            remaining = int(stop_until - current_time)
            cv2.putText(disp, f"WAITING: {remaining}s", (w//4, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            M = cv2.moments(thresh)
            pixel_count = M['m00'] / 255
            Kp, Kd = 45.0, 18.0
            BASE_SPEED, PIVOT_THRESH = 35, 80 

            if 200 < pixel_count < 15000:
                cx = int(M['m10'] / M['m00'])
                error = cx - (w // 2)
                if abs(error) < DEADZONE: error = 0
                last_error = error 
                
                if abs(error) > PIVOT_THRESH:
                    p_speed = 75 
                    set_motors(p_speed if error > 0 else -p_speed, -p_speed if error > 0 else p_speed)
                    steer_filter = 0
                else:
                    norm_err = error / (w / 2)
                    raw_steer = (norm_err * Kp) + ((norm_err - (last_error/(w/2))) * Kd)
                    steer_filter = (SMOOTHING * raw_steer) + ((1 - SMOOTHING) * steer_filter)
                    set_motors(BASE_SPEED + steer_filter, BASE_SPEED - steer_filter)
            else:
                search_speed = 70
                dir_m = 1 if last_error > 0 else -1
                set_motors(search_speed * dir_m, -search_speed * dir_m)

        # 3. ─── UI DASHBOARD ────────────────────────
        cv2.line(disp, (0, line_roi_y), (w, line_roi_y), (255, 0, 0), 2)
        cv2.imshow("Unified Robot Dashboard", disp)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    set_motors(0, 0)
    pwmA.stop(); pwmB.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()

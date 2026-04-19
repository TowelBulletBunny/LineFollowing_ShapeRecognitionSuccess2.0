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
ROI_SPLIT = 0.65     # Use top 65% for signs, bottom 35% for line
N_SCAN = 5           # Scan for shapes every 5 frames to save CPU
MATCH_THRESH = 0.75  # 75% similarity required to trigger a match
DEADZONE = 5         # Ignore steering errors smaller than 5 pixels
SMOOTHING = 0.25     # Percent of new steering value to use (0.0 to 1.0)

# --- GLOBALS ---
last_error = 0
steer_filter = 0
frame_count = 0
stop_until = 0
templates = {}

def load_templates():
    """Reads saved .png images from the folder to use as 'id cards' for matching."""
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
    """Converts speed percentages (-100 to 100) into GPIO signals."""
    l, r = np.clip(l, -100, 100), np.clip(r, -100, 100)
    GPIO.output(IN1, GPIO.LOW if l >= 0 else GPIO.HIGH)
    GPIO.output(IN2, GPIO.HIGH if l >= 0 else GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW if r >= 0 else GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH if r >= 0 else GPIO.LOW)
    pwmA.ChangeDutyCycle(abs(l))
    pwmB.ChangeDutyCycle(abs(r))

def preprocess_template(crop_gray, size=(120, 120)):
    """Prepares a cropped image so it matches the format of our saved templates."""
    # 1. Equalize: Fixes lighting so dark/light parts of the shape are clear
    crop_eq = cv2.equalizeHist(crop_gray)
    
    # 2. Square it: Add padding so resizing doesn't stretch a rectangle into a square
    h, w = crop_eq.shape
    diff = abs(h - w)
    t, b, l, r = (diff//2, diff-diff//2, 0, 0) if h < w else (0, 0, diff//2, diff-diff//2)
    crop_sq = cv2.copyMakeBorder(crop_eq, t, b, l, r, cv2.BORDER_CONSTANT, value=255)
    
    # 3. Final Polish: Add a 10px white border and resize to standard 120x120
    return cv2.resize(cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255), size)

# --- CAMERA INITIALIZATION ---
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
        disp = frame.copy() # Copy for drawing text/lines on UI
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to grayscale for math
        current_time = time.time()

        # 1. ─── SYMBOL DETECTION (Finding Signs) ──────────────────
        # ROI_SPLIT creates a horizontal cut. Top area is for symbols.
        roi_y_top = int(h * ROI_SPLIT)
        sym_roi = gray[0:roi_y_top, :]
        
        # GaussianBlur: Smooths out digital noise (grainy pixels)
        sym_blur = cv2.GaussianBlur(sym_roi, (7, 7), 0)
        
        # Adaptive Threshold: Turns image B&W. Handles uneven lighting (shadows) better than simple threshold.
        sym_bin = cv2.adaptiveThreshold(sym_blur, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 81, 10)
        
        cv2.imshow("Shape Debug (Binary)", sym_bin)

        # Only look for shapes if we aren't currently "paused" by a stop sign
        if current_time > stop_until and frame_count % N_SCAN == 0:
            # FindContours: Detects edges of white blobs in the binary image
            contours, _ = cv2.findContours(sym_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                # Filter by size: Ignore things too small (noise) or too big (the floor)
                if 1500 < area < 30000: 
                    x, y, wb, hb = cv2.boundingRect(c)
                    aspect_ratio = wb / float(hb)
                    
                    # Shapes we want (circles, squares, triangles) are usually roughly 1:1 ratio
                    if 0.5 < aspect_ratio < 1.8 and templates:
                        # Prepare the crop for comparison
                        proc_crop = preprocess_template(sym_roi[y:y+hb, x:x+wb])
                        
                        best_score, best_name = 0, "None"
                        for name, tpl in templates.items():
                            # matchTemplate: Checks how well two images overlap
                            res = cv2.matchTemplate(proc_crop, tpl, cv2.TM_CCOEFF_NORMED)
                            _, score, _, _ = cv2.minMaxLoc(res) # score: 1.0 is identical
                            if score > best_score:
                                best_score, best_name = score, name.split('.')[0]
                        
                        # If a match is strong enough, stop the robot and show the result
                        if best_score > MATCH_THRESH:
                            cur_sym = best_name
                            set_motors(0, 0)
                            stop_until = current_time + 5 # Pause for 5 seconds
                            
                            # UI Popup creation
                            result_win = np.zeros((200, 450, 3), dtype=np.uint8)
                            cv2.putText(result_win, f"SHAPE: {cur_sym}", (20, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(result_win, f"CONF: {int(best_score*100)}%", (20, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                            cv2.imshow("Detection Result", result_win)
                            break

        # 2. ─── MOTOR LOGIC (Line Following) ────────────────────────
        # Look at a narrow horizontal slice of the floor
        line_roi_y = int(h * ROI_SPLIT)
        line_roi = gray[line_roi_y : line_roi_y + 50, :]
        blur = cv2.GaussianBlur(line_roi, (5, 5), 0)
        
        # OTSU Threshold: Automatically finds the best B&W cut-off point for high contrast
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        if ret > 150: thresh = np.zeros_like(thresh) # Safety: if screen is too bright/white, ignore
        cv2.imshow("Line Debug (Threshold)", thresh)

        if current_time < stop_until:
            remaining = int(stop_until - current_time)
            cv2.putText(disp, f"WAITING: {remaining}s", (w//4, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            # Moments: Calculates the weighted center of all white pixels
            M = cv2.moments(thresh)
            pixel_count = M['m00'] / 255
            Kp, Kd = 45.0, 18.0 # Steering sensitivity constants
            BASE_SPEED, PIVOT_THRESH = 35, 80 

            # If we see enough "line" pixels, steer toward them
            if 200 < pixel_count < 15000:
                cx = int(M['m10'] / M['m00']) # X-coordinate of the line center
                error = cx - (w // 2)         # Difference from screen center
                if abs(error) < DEADZONE: error = 0
                
                # Sharp Turn Logic: If the error is massive, pivot in place
                if abs(error) > PIVOT_THRESH:
                    p_speed = 75 
                    set_motors(p_speed if error > 0 else -p_speed, -p_speed if error > 0 else p_speed)
                    steer_filter = 0
                else:
                    # PD Control: Calculate steering based on current error and rate of change
                    norm_err = error / (w / 2)
                    raw_steer = (norm_err * Kp) + ((norm_err - (last_error/(w/2))) * Kd)
                    
                    # Low-pass filter: Prevents jerky wheels by blending old and new steering
                    steer_filter = (SMOOTHING * raw_steer) + ((1 - SMOOTHING) * steer_filter)
                    set_motors(BASE_SPEED + steer_filter, BASE_SPEED - steer_filter)
                
                last_error = error 
            else:
                # Search Mode: If the line is lost, spin in the direction we last saw it
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
    # Cleanup: Turn off motors and close windows
    set_motors(0, 0)
    pwmA.stop(); pwmB.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()

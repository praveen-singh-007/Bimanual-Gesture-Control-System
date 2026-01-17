import math
import cv2
import numpy as np
import HandTrackingModule as htm
import filters.KalmanFilter, filters.OneEuroFilter
from pycaw.pycaw import AudioUtilities
from utils.volume_utils import mcp_angle
from filters import Kalman1D , OneEuroFilter 
from features.volume_class import VolumeControl
import screen_brightness_control as sbc
from features.brightness_class import Brightness
import time
import autopy
from features.cursor_class import Cursor


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()

vc = VolumeControl()
brightness = Brightness()

detector = htm.HandTracking(min_track_conf=0.6, min_detect_conf=0.6)
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
kalman = Kalman1D(process_var=0.05, measurement_var=2.0)
prev_vol = None
prev_brightness = None
# ---------------- CURSOR ----------------
cursor = Cursor()

# ---------------- FILTERS ----------------
filter_x = OneEuroFilter(freq=120, min_cutoff=1.0, beta=0.005)
filter_y = OneEuroFilter(freq=120, min_cutoff=1.0, beta=0.005)

# ---------------- PARAMETERS (UNCHANGED LOGIC) ----------------
velocity_scale = 1.0
pinch_on = 0.35
pinch_off = 0.45
click_debounce = 0.15
double_click_time = 0.35
cursor_update_interval = 1 / 120

REST_TIME = 0.35
REST_SPEED = 0.45
POS_EPS = 1e-3

# ---------------- STATE (PERSISTENT) ----------------
state = dict(
    prev_x=None,
    prev_y=None,
    prev_cursor_x=None,
    prev_cursor_y=None,
    stable_x=None,
    stable_y=None,
    rest_timer=0.0,
    last_filter_time=time.time(),
    last_cursor_update=time.time(),
    last_click_time=0.0,
    pinch_active=False
)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    tracker = detector.find_hands(frame)
    landmarks_list = detector.findPosition(frame, draw=False)
    fingers = detector.finger_up() if landmarks_list else []
    # volume control
    vol_scalar = vc.volume_control(frame, wCam, hCam, device, volume, detector, prev_vol, kalman, landmarks_list)

    # FIRST guard
    if vol_scalar is None:
        cv2.imshow("Volume Control Window", frame)
        continue

    # THEN compare & set
    if prev_vol is None or abs(prev_vol - vol_scalar) > 0.02:
        try:
            volume.SetMasterVolumeLevelScalar(vol_scalar, None)
            prev_vol = vol_scalar
        except Exception as e:
            print("Volume set error:", e)

    #brightness control
    brightness_scalar = brightness.brightness_ctrl(frame, wCam, hCam, detector, kalman, landmarks_list)
    if prev_brightness is None or abs(prev_brightness-brightness_scalar) > 2:
        try:
            
            sbc.set_brightness(int(brightness_scalar), display = 0)


            prev_brightness = brightness_scalar
        except Exception as e:
            print("Brightness set error:", e)

    #cursor control
        # ---------- CURSOR MOVE ----------
    cursor.cursorMove(
        landmarks_list, fingers, frame,
        wCam, hCam, wScr, hScr,
        filter_x, filter_y,
        # adaptive_deadzone,
        velocity_scale,
        pinch_on, pinch_off,
        click_debounce, double_click_time,
        cursor_update_interval,
        REST_TIME, REST_SPEED, POS_EPS,
        state
    )

    # ---------- CURSOR SCROLL ----------
    cursor.cursorScroll(landmarks_list, fingers)
    
    
    cv2.imshow("Volume Control Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


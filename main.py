
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
from features.speech_class import VoiceController
import threading

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = autopy.screen.size()

vc = VolumeControl()
brightness = Brightness()
sr = VoiceController()
# voice_thread = threading.Thread(target = sr.listen, daemon = True)


detector = htm.HandTracking(min_track_conf=0.6, min_detect_conf=0.6)
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
kalman = Kalman1D(process_var=0.05, measurement_var=2.0)
prev_vol = None
prev_brightness = None
voice_active = False

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


# MODE_VOLUME = 0
# MODE_CURSOR = 1
# MODE_BRIGHTNESS = 2

# current_mode = MODE_VOLUME
current_mode = "VR INACTIVE" 
john_cena = [0,0,1,1,1]
thumbs_up = [1,0,0,0,0]
left_hand_gesture_active = False
l_fingers = None
hand_tracking_enabled =True

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



    if hand_tracking_enabled:
        detector.find_hands(frame, True)
    else:
        detector.find_hands(frame, False)  # <-- IMPORTANT

    
    hand_data = []
    left_hand_lm = None
    right_hand_lm = None

    if detector.results and detector.results.multi_hand_landmarks:

        for i in range(len(detector.results.multi_hand_landmarks)):
            landmarks_list, hand_type = detector.findPosition(frame, handNo = i, draw = False)
            if landmarks_list:
                hand_data.append((hand_type, landmarks_list))

    for type, lm in hand_data:
        if type == 'Left':
            left_hand_lm = lm
        elif type == "Right":
            right_hand_lm = lm

    # fingers = detector.finger_up(hand_type) if landmarks_list else []

    if left_hand_lm:
        detector.lmList = left_hand_lm
        left_fingers = detector.finger_up("Left")
        l_fingers = left_fingers

        if left_fingers == thumbs_up:
            hand_tracking_enabled = False
            left_hand_gesture_active = False
            

        if left_fingers == john_cena:
            if not left_hand_gesture_active and not voice_active:
                left_hand_gesture_active = True
                voice_active = True
                sr.active = True
                voice_thread = threading.Thread(
                    target=sr.listen,
                    daemon=True
                )
                voice_thread.start()
        else:
            left_hand_gesture_active = False
    last_mode = current_mode
    current_mode = sr.current_mode

    if sr.last_command_detected:
        hand_tracking_enabled = True
        detector.reset()
        sr.last_command_detected = False


    if current_mode != last_mode:

        if current_mode != "VOLUME":
            prev_vol = None

        if current_mode != "BRIGHTNESS":
            prev_brightness = None

        # if current_mode != "VOLUME":
        #     prev_vol = None

        # if current_mode != "BRIGHTNESS":
        #     prev_brightness = None
        

    if right_hand_lm and current_mode in ["VOLUME","BRIGHTNESS","CURSOR"]:
        detector.lmList = right_hand_lm
        right_fingers = detector.finger_up("Right")
    
        if current_mode == "VOLUME":
            # # volume control
            vol_scalar = vc.volume_control(frame, wCam, hCam, device, volume, detector, prev_vol, kalman, right_hand_lm)

            # FIRST guard
            if vol_scalar is None:
                cv2.imshow("Volume Control Window", frame)
                continue

            # # THEN compare & set
            if prev_vol is None or abs(prev_vol - vol_scalar) > 0.02:
                try:
                    volume.SetMasterVolumeLevelScalar(vol_scalar, None)
                    prev_vol = vol_scalar
                except Exception as e:
                    print("Volume set error:", e)

            #brightness control
        elif current_mode == "BRIGHTNESS":
            brightness_scalar = brightness.brightness_ctrl(frame, wCam, hCam, detector, kalman, right_hand_lm)
            if prev_brightness is None or abs(prev_brightness-brightness_scalar) > 2:
                try:
                    sbc.set_brightness(int(brightness_scalar), display = 0)
                    prev_brightness = brightness_scalar
                except Exception as e:
                    print("Brightness set error:", e)

        elif current_mode == "CURSOR":
            # ---------- CURSOR MOVE ----------
            cursor.cursorMove(
                right_hand_lm, right_fingers, frame,
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
            cursor.cursorScroll(right_hand_lm, right_fingers)

    # # set changes
    # if l_fingers == thumbs_up:
    #     hand_tracking_enabled = False
    #     left_hand_gesture_active = False

    cv2.putText(frame, f"MODE: {current_mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Volume Control Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



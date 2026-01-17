import cv2
import HandTrackingModule as htm
import subprocess
import time

detector = htm.HandTracking(min_detect_conf=0.6, min_track_conf=0.6, num_hands=1)
cap = cv2.VideoCapture(0)

gesture_timeout = 2.0
cool_down = 5.0

gesture_count = 0
last_gesture_time = 0
last_launch_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector.find_hands(frame, draw=True)
    landmark_list = detector.findPosition(frame, draw=False)

    gesture_active = False  

    if len(landmark_list) != 0:
        middle_tip = landmark_list[12][2]
        ring_tip = landmark_list[16][2]
        index_pip = landmark_list[7][2]
        pinky_pip = landmark_list[19][2]

        fingers = detector.finger_up()

        if fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1:
            if middle_tip > index_pip and ring_tip > pinky_pip:
                gesture_active = True

    now = time.time()

    if gesture_active and now - last_launch_time > cool_down:

        if gesture_count == 0:
            gesture_count = 1
            last_gesture_time = now
            print("Gesture detected once")

        elif gesture_count == 1 and (now - last_gesture_time) < gesture_timeout:
            print("Incognito triggered")

            subprocess.Popen([
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "--incognito"
            ])

            last_launch_time = now
            gesture_count = 0


    if gesture_count == 1 and (
        not gesture_active or (now - last_gesture_time) > gesture_timeout
    ):
        gesture_count = 0

    cv2.imshow("Incognito Window", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

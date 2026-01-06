import cv2
import HandTrackingModule as htm
import numpy as np
import autopy
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
frameR = 150
wScr, hScr = autopy.screen.size()
detector = htm.HandTracking(min_detect_conf=0.6, min_track_conf=0.6, num_hands=1)
while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    tracker = detector.find_hands(frame, draw=True)
    landmarks_list = detector.findPosition(frame, draw=False)

    if len(landmarks_list)!=0:
        thumb_tip = (landmarks_list[4][1], landmarks_list[4][2])
        middle_tip = (landmarks_list[12][1], landmarks_list[12][2])
        fingers = detector.finger_up()

        cv2.rectangle(frame, (frameR,frameR),(wCam-frameR, hCam - frameR), (0,0,255),1)
        if fingers[0] == 1 and fingers[2] == 0:

            x = np.interp(thumb_tip[0],(frameR, wCam-frameR),(0,wScr))
            y = np.interp(thumb_tip[1],(frameR,hCam-frameR),(0,hScr))

            # x = wScr - x


            x = np.clip(x, 0, wScr - 1)
            y = np.clip(y, 0, hScr - 1)


            autopy.mouse.move(x,y)

    cv2.imshow("Cursor Control Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


#2. Get the tip of the index and middle finger
#3. Check which fingers are up
#4. Only Index Finger Moving Mode
#5. Convert Coordinates
#6. Smoothen Values
#7. Move Mouse
#8. Both Index and middle fingers are up: Clicking Mode
#9. Find distance between fingers
#10. Click mouse if distance short
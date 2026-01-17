import math
import cv2
import numpy as np
import HandTrackingModule as htm
import filters.KalmanFilter as km
import screen_brightness_control as sbc
from utils.volume_utils import mcp_angle
# brightness = sbc.set_brightness()
# print(brightness)
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandTracking(min_track_conf=0.6, min_detect_conf=0.6)
tip_ids = [4,8]
prev_brightness = None
min_angle = 15
max_angle = 65
kalman = km.Kalman1D(
    process_var=0.05,
    measurement_var=2.0
)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break

    tracker = detector.find_hands(frame)
    landmarks_list = detector.findPosition(frame, draw=False)



  

    if len(landmarks_list)!=0:
        x1, y1 = landmarks_list[4][1], landmarks_list[4][2] 
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
        cx, cy = int((x1+x2)//2) , int((y1+y2)//2) 
        cv2.line(frame, (x1,y1),(x2,y2), (0,0,225), 3)
        cv2.circle(img = frame, center =(cx, cy), color = (0,0,255), thickness=-1, radius = 7) 
        cv2.circle(img = frame, center =(x1, y1), color = (0,0,255), thickness=-1, radius = 9) 
        cv2.circle(img = frame, center =(x2, y2), color = (0,0,255), thickness=-1, radius = 9)
        index_tip = (landmarks_list[8][1],landmarks_list[8][2])
        index_mcp = (landmarks_list[5][1],landmarks_list[5][2])
        thumb_tip = (landmarks_list[4][1],landmarks_list[4][2])
        raw_angle = mcp_angle(index_tip,index_mcp,thumb_tip)
        if raw_angle is None:
            continue
        smoothened_angle = kalman.update(raw_angle)


        angle = np.clip(smoothened_angle, min_angle, max_angle)
       
        brightness_scalar = np.interp(angle,[min_angle,max_angle],[0,100])

        brightness_scalar = float(np.clip(brightness_scalar, 0,100))

        
        if prev_brightness is None or abs(prev_brightness-brightness_scalar) > 2:
            try:
                
                sbc.set_brightness(int(brightness_scalar), display = 0)


                prev_brightness = brightness_scalar
            except Exception as e:
                print("Brightness set error:", e)


        cv2.putText(frame, f"Angle: {int(angle)}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),2)

    cv2.imshow("Brightness Control Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
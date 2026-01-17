import math
import cv2
import numpy as np
import HandTrackingModule as htm
import filters.KalmanFilter as km
import screen_brightness_control as sbc
from utils.volume_utils import mcp_angle


class Brightness():

    def __init__(self):
        pass

    def brightness_ctrl(self, frame, wCam, hCam, detector, kalman, landmarks_list):
        
        min_angle = 15
        max_angle = 65

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
    
            smoothened_angle = kalman.update(raw_angle)


            angle = np.clip(smoothened_angle, min_angle, max_angle)
        
            brightness_scalar = np.interp(angle,[min_angle,max_angle],[0,100])

            brightness_scalar = float(np.clip(brightness_scalar, 0,100))

            return brightness_scalar
        
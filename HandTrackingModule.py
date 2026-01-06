import cv2
import mediapipe as mp


# make class and constructor with Hands() param
# make find_hands fx to detect hands and draw lines, with draw param
# def fx to find position and print co-ordinates


class HandTracking():

    def __init__(self, min_track_conf=0.5, min_detect_conf = 0.5, num_hands = 2, model_complex=0, mode=False):
        self.mode = mode
        self.num_hands = num_hands
        self.model_complex = model_complex
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf
        self.draw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.model_complex, self.min_detect_conf, self.min_track_conf)
        self.tip_ids = [4,8,12,16,20]

    def find_hands(self, frame, draw = True):

       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(frame, landmark, self.mpHands.HAND_CONNECTIONS)

    def findPosition(self, frame, handNo=0, draw=True):

        self.lmList =[]

        if self.results.multi_hand_landmarks:
            handLm = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(handLm.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def finger_up(self):

        fingers = []

        if self.lmList[self.tip_ids[0]][1] > self.lmList[self.tip_ids[0]-1][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        
        for i in range(1,5):
            if self.lmList[self.tip_ids[i]][2] < self.lmList[self.tip_ids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
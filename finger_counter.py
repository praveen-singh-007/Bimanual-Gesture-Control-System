import cv2
import os
import HandTrackingModule as htm

folder_path = "img_path"
my_list = os.listdir("img_path")

overlay_list = []
for img_path in my_list:
    path = cv2.imread(f"{folder_path}/{img_path}")
    overlay_list.append(path)

detector = htm.HandTracking(min_track_conf=0.7)
tip_ids = [4,8,12,16,20]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    tracker = detector.find_hands(frame)
    landmark_list = detector.findPosition(frame, draw=False)

    if len(landmark_list) != 0:
        fingers = []

        if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0]-1][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        
        for i in range(1,5):
            if landmark_list[tip_ids[i]][2] < landmark_list[tip_ids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)

        h, w, c = overlay_list[total_fingers-1].shape

        frame[0:h,0:w] = overlay_list[total_fingers-1]
    






    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

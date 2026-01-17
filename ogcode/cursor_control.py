
import cv2
import HandTrackingModule as htm
import numpy as np
import autopy
# import KalmanFilter as km
import filters.OneEuroFilter as OEF
import time
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
frameR = 150
wScr, hScr = autopy.screen.size()
detector = htm.HandTracking(min_detect_conf=0.6, min_track_conf=0.6, num_hands=1)

filter_x = OEF.OneEuroFilter(freq = 120, min_cutoff= 1.4, beta= 0.01)
filter_y = OEF.OneEuroFilter(freq = 120, min_cutoff= 1.4, beta= 0.01)
# kalman_x = km.Kalman1D(process_var=0.08, measurement_var=1.0)
# kalman_y = km.Kalman1D(process_var=0.08, measurement_var=1.0)
pinch_active = False
prev_x, prev_y = None, None
prev_cursor_x, prev_cursor_y = None, None
pinch_on = 0.28
pinch_off = 0.42
double_click_time = 0.45
cursor_update_interval = 1/60
last_click_time = 0
click_debounce = 0.15
last_cursor_update = 0
velocity_scale = 0.85

rest_timer = 0.0
REST_SPEED = 0.08    # below this → candidate for rest
REST_TIME = 0.25       # seconds needed to enter rest

last_filter_time = time.time()
POS_EPS = 0.8   # pixels in screen space

stable_x = None
stable_y = None

def adaptive_deadzone(v, speed, base=0.4, min_dz=0.12):
    dz = max(min_dz, base - speed * 0.08)
    if abs(v) <= dz:
        return 0.0
    return np.sign(v) * (abs(v) - dz)



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if not ret:
        break
    
    tracker = detector.find_hands(frame, draw=True)
    landmarks_list = detector.findPosition(frame, draw=False)

    # detector.find_hands(frame_display, draw=True)

    if len(landmarks_list)!=0:
        thumb_tip = (landmarks_list[4][1], landmarks_list[4][2])
        index_tip = (landmarks_list[8][1], landmarks_list[8][2])
        fingers = detector.finger_up()
        # length = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] + index_tip[1])**2)**0.5
        dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        # print(length, dist)
        index_mcp = np.array((landmarks_list[5][1], landmarks_list[5][2]))
        pinky_mcp = (landmarks_list[17][1], landmarks_list[17][2])
        middle_mcp = np.array((landmarks_list[9][1], landmarks_list[9][2]))
        wrist = np.array((landmarks_list[0][1], landmarks_list[0][2]))

        cursor_point = (index_mcp + middle_mcp + wrist)/3

        normalized_hand = np.linalg.norm(np.array(index_mcp)-np.array(pinky_mcp))

        if normalized_hand == 0:
            continue

        pinch_ratio = dist/normalized_hand

        cv2.rectangle(frame, (frameR,frameR),(wCam-frameR, hCam - frameR), (0,0,255),1)
        if fingers[0] == 1 and fingers[2:] == [0,0,0]:

            x = np.interp(cursor_point[0],(frameR, wCam-frameR),(0,wScr))
            y = np.interp(cursor_point[1],(frameR,hCam-frameR),(0,hScr))

            #x = wScr - x


            x = np.clip(x, 0, wScr - 1)
            y = np.clip(y, 0, hScr - 1)
            now = time.time()
            dt = max(now - last_filter_time, 1e-3)
            dt = np.clip(dt, 1/240, 1/30)

            last_filter_time = now

            filter_x.freq = 1.0 / dt
            filter_y.freq = 1.0 / dt

            if rest_timer >= REST_TIME and stable_x is not None:
                smooth_x = stable_x
                smooth_y = stable_y
            else:
                smooth_x = filter_x.update(x)
                smooth_y = filter_y.update(y)



            
            if prev_x == None:
                prev_x, prev_y = smooth_x, smooth_y
                prev_cursor_x, prev_cursor_y = smooth_x, smooth_y
                continue

            # if abs(smooth_x - prev_x) < POS_EPS and abs(smooth_y - prev_y) < POS_EPS:
            #     smooth_x = prev_x
            #     smooth_y = prev_y


            dx = (smooth_x - prev_x) / dt
            dy = (smooth_y - prev_y) / dt


            speed = np.hypot(dx, dy)

            dx = adaptive_deadzone(dx, speed)
            dy = adaptive_deadzone(dy, speed)


            speed = np.hypot(dx, dy)

            if speed < REST_SPEED:
                rest_timer += dt
                if rest_timer >= REST_TIME:
                    dx = 0.0
                    dy = 0.0
                    # LOCK FILTER OUTPUT
                    stable_x = prev_x
                    stable_y = prev_y
                    # HARD FREEZE
                    prev_cursor_x = prev_cursor_x
                    prev_cursor_y = prev_cursor_y
                    prev_x = smooth_x
                    prev_y = smooth_y
            else:
                rest_timer = 0.0
                stable_x = None
                stable_y = None


            if rest_timer < REST_TIME and speed < 0.8:
                dx *= 0.35
                dy *= 0.35
            elif rest_timer >= REST_TIME:
                dx = 0.0
                dy = 0.0



            # speed =  np.hypot(dx,dy)
            # #dead-zone
            # if speed < 0.3:
            #     dx = dy = 0
            # #axis stabelization
            # if abs(dx) < 1.5:
            #     dx = 0
            # if abs(dy) < 1.5:
            #     dy = 0
            # #stable gain
            # if speed < 2.0:
            #     gain = 1.0
            # else:
            #     gain = 1.025 + min((speed/16),1.0)*1.0
            now = time.time()
            if now - last_cursor_update >= cursor_update_interval:
        
                last_cursor_update = now

                accel = 1.0 + min(speed / 6.0, 1.0) * 0.6
                vx = velocity_scale * dx * accel * dt
                vy = velocity_scale * dy * accel * dt 

                cx = prev_cursor_x + vx
                cy = prev_cursor_y + vy

                cx = np.clip(cx,0,wScr-1)
                cy = np.clip(cy,0,hScr-1)
                autopy.mouse.move(round(cx), round(cy))

                if rest_timer < REST_TIME:
                    prev_x, prev_y = smooth_x, smooth_y
                    prev_cursor_x, prev_cursor_y = cx, cy


            if pinch_ratio < pinch_on and not pinch_active:
                now = time.time()

                # if now - last_click_time < cursor_update_interval:
                #     continue
                # last_click_time = now   

                if now - last_click_time < click_debounce:
                    pass
                else:
                    if now - last_click_time < double_click_time:
                        autopy.mouse.click(autopy.mouse.Button.LEFT, 2)
                        last_click_time = 0
                    else:
                        autopy.mouse.click()
                        last_click_time = now
                pinch_active = True
            elif pinch_ratio > pinch_off:
                pinch_active = False
            

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
import cv2
import numpy as np
import pyautogui
import time
from utils.adaptive_deadzone import adaptive_deadzone

class Cursor:
    def __init__(self):
        self.scroll_active = False
        self.prev_scroll_y = 0
        self.scroll_sensitivity = 0.8

    # def cursorMove(self, landmarks, fingers, frame, wCam, hCam, wScr, hScr):
        # if not landmarks:
        #     return

        # frameR = 150
        # thumb_tip = (landmarks[4][1], landmarks[4][2])

        # if fingers[0] == 1 and fingers[2] == 0:
        #     x = np.interp(thumb_tip[0], (frameR, wCam - frameR), (0, wScr))
        #     y = np.interp(thumb_tip[1], (frameR, hCam - frameR), (0, hScr))

        #     x = np.clip(x, 0, wScr - 1)
        #     y = np.clip(y, 0, hScr - 1)

        #     pyautogui.moveTo(int(x), int(y))
        #     cv2.circle(frame, thumb_tip, 12, (255, 0, 255), cv2.FILLED)


    def cursorMove(self, landmarks, fingers, frame,
                wCam, hCam, wScr, hScr,
                filter_x, filter_y,
                # adaptive_deadzone,
                velocity_scale,
                pinch_on, pinch_off,
                click_debounce, double_click_time,
                cursor_update_interval,
                REST_TIME, REST_SPEED, POS_EPS,
                state):
        """
        state: dict holding mutable state variables
        """

        if not landmarks:
            return

        frameR = 150

        thumb_tip = np.array((landmarks[4][1], landmarks[4][2]))
        index_tip = np.array((landmarks[8][1], landmarks[8][2]))

        dist = np.linalg.norm(thumb_tip - index_tip)

        index_mcp = np.array((landmarks[5][1], landmarks[5][2]))
        pinky_mcp = np.array((landmarks[17][1], landmarks[17][2]))
        middle_mcp = np.array((landmarks[9][1], landmarks[9][2]))
        wrist = np.array((landmarks[0][1], landmarks[0][2]))

        cursor_point = (index_mcp + middle_mcp + wrist) / 3.0
        normalized_hand = np.linalg.norm(index_mcp - pinky_mcp)

        if normalized_hand == 0:
            return

        pinch_ratio = dist / normalized_hand

        cv2.rectangle(frame,
                    (frameR, frameR),
                    (wCam - frameR, hCam - frameR),
                    (0, 0, 255), 1)

        # ===== MOVE MODE =====
        if fingers[0] == 1 and fingers[2:] == [0, 0, 0]:

            x = np.interp(cursor_point[0], (frameR, wCam - frameR), (0, wScr))
            y = np.interp(cursor_point[1], (frameR, hCam - frameR), (0, hScr))

            x = np.clip(x, 0, wScr - 1)
            y = np.clip(y, 0, hScr - 1)

            now = time.time()
            dt = max(now - state["last_filter_time"], 1e-3)
            dt = np.clip(dt, 1 / 240, 1 / 30)
            state["last_filter_time"] = now

            filter_x.freq = 1.0 / dt
            filter_y.freq = 1.0 / dt

            if state["rest_timer"] >= REST_TIME and state["stable_x"] is not None:
                smooth_x = state["stable_x"]
                smooth_y = state["stable_y"]
            else:
                smooth_x = filter_x.update(x)
                smooth_y = filter_y.update(y)

            if state["prev_x"] is None:
                state["prev_x"] = smooth_x
                state["prev_y"] = smooth_y
                state["prev_cursor_x"] = smooth_x
                state["prev_cursor_y"] = smooth_y
                return

            dx = (smooth_x - state["prev_x"]) / dt
            dy = (smooth_y - state["prev_y"]) / dt

            speed = np.hypot(dx, dy)

            dx = adaptive_deadzone(dx, speed)
            dy = adaptive_deadzone(dy, speed)

            speed = np.hypot(dx, dy)

            if speed < REST_SPEED:
                state["rest_timer"] += dt
                if state["rest_timer"] >= REST_TIME:
                    dx = dy = 0.0
                    state["stable_x"] = state["prev_x"]
                    state["stable_y"] = state["prev_y"]
            else:
                state["rest_timer"] = 0.0
                state["stable_x"] = None
                state["stable_y"] = None

            if state["rest_timer"] < REST_TIME and speed < 0.8:
                dx *= 0.35
                dy *= 0.35
            elif state["rest_timer"] >= REST_TIME:
                dx = dy = 0.0

            now = time.time()
            if now - state["last_cursor_update"] >= cursor_update_interval:
                state["last_cursor_update"] = now

                accel = 1.0 + min(speed / 6.0, 1.0) * 0.6
                vx = velocity_scale * dx * accel * dt
                vy = velocity_scale * dy * accel * dt

                cx = state["prev_cursor_x"] + vx
                cy = state["prev_cursor_y"] + vy

                cx = np.clip(cx, 0, wScr - 1)
                cy = np.clip(cy, 0, hScr - 1)

                pyautogui.moveTo(round(cx), round(cy))

                if state["rest_timer"] < REST_TIME:
                    state["prev_x"] = smooth_x
                    state["prev_y"] = smooth_y
                    state["prev_cursor_x"] = cx
                    state["prev_cursor_y"] = cy

        # ===== CLICK / DOUBLE CLICK =====
        now = time.time()

        if pinch_ratio < pinch_on and not state["pinch_active"]:
            if now - state["last_click_time"] >= click_debounce:
                if now - state["last_click_time"] < double_click_time:
                    pyautogui.click(button= 'left')
                    state["last_click_time"] = 0
                else:
                    pyautogui.click()
                    state["last_click_time"] = now
            state["pinch_active"] = True

        elif pinch_ratio > pinch_off:
            state["pinch_active"] = False




    def cursorScroll(self, landmarks, fingers):
        if not landmarks:
            self.scroll_active = False
            return

        # index finger tip Y (camera space)
        y_index = landmarks[8][2]
        y_middle = landmarks[12][2]

        # index + middle up, others down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
            if not self.scroll_active:
                self.scroll_active = True
                self.prev_scroll_y_index = y_index
                self.prev_scroll_y_middle = y_middle
            else:
                dy_index = y_index - self.prev_scroll_y_index
                dy_middle = y_middle - self.prev_scroll_y_middle
                self.prev_scroll_y_index = y_index
                self.prev_scroll_y_middle= y_middle

                if abs(dy_index) > 8:
                    pyautogui.vscroll(int(-80))

                if abs(dy_middle) > 8:
                    pyautogui.vscroll(int(60))
        else:
            self.scroll_active = False
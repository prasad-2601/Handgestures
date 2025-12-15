import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
tip_ids = [4, 8, 12, 16, 20]
last_click_time = 0
click_delay = 0.4   # seconds

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((id, int(lm.x * w), int(lm.y * h)))

            fingers = [0]  # ignore thumb

            for i in range(1, 5):
                if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_x, index_y = lm_list[8][1], lm_list[8][2]
            middle_x, middle_y = lm_list[12][1], lm_list[12][2]

            # Move mouse
            if fingers[1] == 1 and fingers[2] == 0:
                screen_x = np.interp(index_x, (0, w), (0, screen_w))
                screen_y = np.interp(index_y, (0, h), (0, screen_h))
                pyautogui.moveTo(screen_x, screen_y)

            # Distance between index & middle finger
            distance = math.hypot(middle_x - index_x, middle_y - index_y)

            current_time = time.time()

            # Left Click
            if fingers[1] == 1 and fingers[2] == 1 and distance < 30:
                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame, "Left Click", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click
            if fingers[1] == 0 and fingers[2] == 1:
                if current_time - last_click_time > click_delay:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    cv2.putText(frame, "Right Click", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
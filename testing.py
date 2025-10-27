import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from utils import get_landmarks_as_row

gesture_name = input("Enter gesture name: ")
csv_filename = "gestures.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
data = []

print("Press 's' to save sample, 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = get_landmarks_as_row(hand_landmarks.landmark)
            if row:
                cv2.putText(frame, gesture_name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Training Gesture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and results.multi_hand_landmarks:
        data.append([gesture_name] + row)
        print(f"Saved sample {len(data)} for {gesture_name}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv(csv_filename, mode='a', header=not pd.io.common.file_exists(csv_filename), index=False)
print(f"Saved {len(data)} samples to {csv_filename}")

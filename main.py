import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyttsx3
from display import OledDisplay
from utils import calculate_distance, normalize_landmarks, get_landmarks_as_row

# Initialize OLED Display
oled = OledDisplay()
oled.startup_animation("TARANG")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load gestures
csv_filename = "gestures.csv"
df = pd.read_csv(csv_filename) if csv_filename else None

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Function to find closest gesture
def find_closest_gesture(landmarks):
    if df is None or df.empty:
        return "No gestures found."
    input_row = np.array(get_landmarks_as_row(landmarks))
    if input_row is None:
        return "Invalid landmarks."
    best_match, min_distance = None, float("inf")
    for _, row in df.iterrows():
        stored_row = np.array(row[1:].values, dtype=np.float32)
        distance = np.linalg.norm(input_row - stored_row)
        if distance < min_distance:
            min_distance = distance
            best_match = row["gesture"]
    return best_match or "Unknown Gesture"

# Start camera
cap = cv2.VideoCapture(0)
prev_gesture = None
oled.show_text("System Ready...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = find_closest_gesture(landmarks.landmark)

            cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if gesture != prev_gesture:
                oled.show_text(f"{gesture}")
                print(f"Recognized: {gesture}")
                engine.say(gesture)
                engine.runAndWait()
                prev_gesture = gesture

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

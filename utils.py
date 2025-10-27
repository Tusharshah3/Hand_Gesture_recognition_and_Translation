import numpy as np

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]))

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    middle = landmarks[9]
    scale = calculate_distance(wrist, middle)
    if scale == 0:
        return None
    normalized = []
    for lm in landmarks:
        normalized.append([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])
    return normalized

def get_landmarks_as_row(landmarks):
    normalized = normalize_landmarks(landmarks)
    if not normalized:
        return None
    return [coord for lm in normalized for coord in lm]

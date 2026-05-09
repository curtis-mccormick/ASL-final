import cv2
import json
import os
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = os.path.join("models", "asl_model.keras")
LABELS_PATH = os.path.join("models", "label_classes.json")


# -----------------------------
# Settings
# -----------------------------
MAX_NUM_HANDS = 1
CONFIDENCE_THRESHOLD = 0.70

# Smooth predictions so it does not flicker every frame
SMOOTHING_WINDOW = 10


# -----------------------------
# Load model and labels
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model file: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Could not find labels file: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

print("Loaded model:", MODEL_PATH)
print("Loaded labels:", labels)


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


prediction_history = deque(maxlen=SMOOTHING_WINDOW)

prev_time = time.time()


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        predicted_label = "No hand"
        predicted_confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Convert landmarks into one flat input row:
            # x0, y0, z0, x1, y1, z1, ... x20, y20, z20
            landmark_row = []

            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])

            X = np.array([landmark_row], dtype=np.float32)

            prediction = model.predict(X, verbose=0)[0]

            class_index = int(np.argmax(prediction))
            confidence = float(prediction[class_index])
            raw_label = labels[class_index]

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(raw_label)
            else:
                prediction_history.append("?")

            # Use the most common prediction from recent frames
            most_common_label, count = Counter(prediction_history).most_common(1)[0]

            predicted_label = most_common_label
            predicted_confidence = confidence

        else:
            prediction_history.clear()

        # -----------------------------
        # FPS calculation
        # -----------------------------
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        # -----------------------------
        # Draw display text
        # -----------------------------
        cv2.rectangle(frame, (0, 0), (420, 150), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"Prediction: {predicted_label}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {predicted_confidence:.2f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("ASL Live Prediction", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
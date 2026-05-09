import cv2
import csv
import os
import time
import mediapipe as mp

# -----------------------------
# Settings
# -----------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "asl_landmarks.csv")

MAX_NUM_HANDS = 1  # Keep this as 1 for the first version
VALID_LABELS = list("ABCDE")  # Start small first

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Create CSV header if needed
# -----------------------------
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)

        header = ["label"]

        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]

        writer.writerow(header)

print(f"Saving data to: {CSV_PATH}")
print("Controls:")
print("  Press A, B, C, D, or E to save one sample")
print("  Press Q to quit")

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

sample_count = 0
last_saved_label = None
last_saved_time = 0

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

        hand_detected = False
        landmark_row = None

        if results.multi_hand_landmarks:
            hand_detected = True

            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Convert landmarks into one flat row:
            # x0, y0, z0, x1, y1, z1, ...
            landmark_row = []

            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])

        # -----------------------------
        # On-screen instructions
        # -----------------------------
        cv2.putText(
            frame,
            "Hold ASL sign, press A/B/C/D/E to save",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples saved this session: {sample_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if hand_detected:
            cv2.putText(
                frame,
                "Hand detected",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        if last_saved_label is not None:
            cv2.putText(
                frame,
                f"Last saved: {last_saved_label}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        key_char = chr(key).upper() if key != 255 else ""

        if key_char in VALID_LABELS:
            if landmark_row is None:
                print(f"No hand detected. Could not save label {key_char}.")
                continue

            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([key_char] + landmark_row)

            sample_count += 1
            last_saved_label = key_char
            last_saved_time = time.time()

            print(f"Saved sample {sample_count}: {key_char}")

cap.release()
cv2.destroyAllWindows()
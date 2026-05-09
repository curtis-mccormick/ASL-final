import cv2
import csv
import os
import time
import mediapipe as mp
from collections import Counter


# -----------------------------
# Settings
# -----------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "asl_landmarks.csv")

MAX_NUM_HANDS = 1

# Static ASL letters only.
# J and Z are skipped because they require motion.
VALID_LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# Landmark normalization
# -----------------------------
def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe hand landmarks into a normalized 63-value feature vector.

    This makes the model care more about hand shape and less about:
    - where your hand is on the screen
    - how close your hand is to the camera

    Process:
    1. Use the wrist landmark as the origin.
    2. Subtract wrist x/y/z from every point.
    3. Scale all points by the largest distance from the wrist.
    """

    points = []

    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])

    wrist = points[0]

    normalized_points = []
    max_distance = 0

    for p in points:
        x = p[0] - wrist[0]
        y = p[1] - wrist[1]
        z = p[2] - wrist[2]

        distance = (x**2 + y**2 + z**2) ** 0.5

        if distance > max_distance:
            max_distance = distance

        normalized_points.append([x, y, z])

    if max_distance == 0:
        max_distance = 1

    feature_vector = []

    for p in normalized_points:
        feature_vector.extend([
            p[0] / max_distance,
            p[1] / max_distance,
            p[2] / max_distance
        ])

    return feature_vector


# -----------------------------
# CSV helpers
# -----------------------------
def create_csv_if_needed():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)

            header = ["label"]

            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]

            writer.writerow(header)


def load_label_counts():
    counts = Counter()

    if not os.path.exists(CSV_PATH):
        return counts

    with open(CSV_PATH, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) > 0:
                label = row[0]
                counts[label] += 1

    return counts


def append_sample(label, landmark_row):
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + landmark_row)


def undo_last_sample():
    """
    Deletes the most recent saved sample from the CSV.
    Returns the label that was deleted, or None if nothing was deleted.
    """

    if not os.path.exists(CSV_PATH):
        return None

    with open(CSV_PATH, mode="r", newline="") as f:
        rows = list(csv.reader(f))

    # rows[0] is the header. If only header exists, there is nothing to undo.
    if len(rows) <= 1:
        return None

    deleted_row = rows.pop()
    deleted_label = deleted_row[0]

    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return deleted_label


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# Startup
# -----------------------------
create_csv_if_needed()
label_counts = load_label_counts()

print(f"Saving data to: {CSV_PATH}")
print()
print("Controls:")
print("  Press a letter key to save that label.")
print("  Valid labels:", "".join(VALID_LABELS))
print("  Backspace = undo last saved sample")
print("  ESC = quit")
print()
print("Current counts:")
for label in VALID_LABELS:
    print(f"  {label}: {label_counts[label]}")


# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


session_count = 0
last_saved_label = None
last_message = "Ready"
last_message_time = time.time()


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

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_row = normalize_landmarks(hand_landmarks)

        # -----------------------------
        # Display background panel
        # -----------------------------
        # -----------------------------
        # Compact display overlay
        # -----------------------------
        overlay_height = 115
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "ASL Data Collector",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Letter = save | Backspace = undo | ESC = quit",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        status_text = "Hand detected" if hand_detected else "No hand"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"Session: {session_count}",
            (200, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            last_message,
            (350, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # -----------------------------
        # Small count display at bottom
        # -----------------------------
        counts_text = "  ".join([f"{label}:{label_counts[label]}" for label in VALID_LABELS])

        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 35),
            (frame.shape[1], frame.shape[0]),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            counts_text[:95],
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # Escape to quit
        if key == 27:
            break

        # Backspace to undo last sample
        # Backspace is usually 8 on Windows/OpenCV.
        if key == 8:
            deleted_label = undo_last_sample()

            if deleted_label is None:
                last_message = "Nothing to undo"
                print("Nothing to undo.")
            else:
                label_counts[deleted_label] -= 1
                session_count = max(0, session_count - 1)
                last_saved_label = None
                last_message = f"Undid last sample: {deleted_label}"
                print(f"Undid last sample: {deleted_label}")

            last_message_time = time.time()
            continue

        # Save sample with letter key
        key_char = chr(key).upper() if key != 255 else ""

        if key_char in VALID_LABELS:
            if landmark_row is None:
                last_message = f"No hand detected. Could not save {key_char}."
                print(last_message)
                last_message_time = time.time()
                continue

            append_sample(key_char, landmark_row)

            label_counts[key_char] += 1
            session_count += 1
            last_saved_label = key_char
            last_message = f"Saved {key_char} | Total {key_char}: {label_counts[key_char]}"
            last_message_time = time.time()

            print(last_message)


cap.release()
cv2.destroyAllWindows()
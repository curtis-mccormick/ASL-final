import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = os.path.join("data", "asl_landmarks.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Loaded dataset:")
print(df.head())
print()
print("Class counts:")
print(df["label"].value_counts())
print()

# Labels: A, B, C, D, E
y_labels = df["label"].values

# Features: x0, y0, z0, ..., x20, y20, z20
X = df.drop("label", axis=1).values.astype("float32")


# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

num_classes = len(label_encoder.classes_)

print("Classes:")
print(label_encoder.classes_)
print()


# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Build model
# -----------------------------
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.25),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Train
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print()
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print()

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# Save model and labels
# -----------------------------
model.save(MODEL_PATH)

with open(LABELS_PATH, "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

print()
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved labels to: {LABELS_PATH}")
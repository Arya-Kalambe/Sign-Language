import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from word_config import (
    WORDS,
    WORD_DATA_PATH,
    WORD_MODEL_PATH,
    WORD_LABELS_PATH,
)
from config import SEQUENCE_LENGTH, FEATURES_PER_FRAME


def load_dataset():
    X = []
    y = []

    for label_idx, word in enumerate(WORDS):
        word_dir = WORD_DATA_PATH / word

        if not word_dir.exists():
            continue

        for seq_dir in sorted(
            [p for p in word_dir.iterdir() if p.is_dir()]
        ):
            frames = []
            valid = True

            for i in range(SEQUENCE_LENGTH):
                p = seq_dir / f"{i}.npy"

                if not p.exists():
                    valid = False
                    break

                arr = np.load(p).astype("float32")

                if arr.shape != (FEATURES_PER_FRAME,):
                    valid = False
                    break

                frames.append(arr)

            if valid:
                X.append(frames)
                y.append(label_idx)

    if not X:
        raise RuntimeError(
            f"No valid word sequences found in {WORD_DATA_PATH}. "
            "Run collect_words.py first."
        )

    return (
        np.asarray(X, dtype="float32"),
        np.asarray(y, dtype="int64"),
    )


X, y = load_dataset()

present = sorted(set(y.tolist()))

print("Dataset:", X.shape)
print("Classes present:", [WORDS[i] for i in present])

if len(present) < 2:
    raise RuntimeError("Need at least 2 word classes to train.")

# A stratified 15% test split requires enough examples in every class.
counts = np.bincount(y)
if np.min(counts[counts > 0]) < 2:
    raise RuntimeError(
        "Each word needs at least 2 sequences for a stratified split. "
        "Collect more data first."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

y_train_cat = keras.utils.to_categorical(
    y_train,
    num_classes=len(WORDS),
)
y_test_cat = keras.utils.to_categorical(
    y_test,
    num_classes=len(WORDS),
)

model = keras.Sequential([
    layers.Input(
        shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    ),

    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),

    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),

    layers.LSTM(64),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(32, activation="relu"),

    layers.Dense(
        len(WORDS),
        activation="softmax",
    ),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-5,
    ),
]

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)

loss, acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0,
)

print(f"Word model test accuracy: {acc * 100:.2f}%")

model.save(WORD_MODEL_PATH)
np.save(WORD_LABELS_PATH, WORDS)

print(f"Saved word model: {WORD_MODEL_PATH}")
print(f"Saved word labels: {WORD_LABELS_PATH}")

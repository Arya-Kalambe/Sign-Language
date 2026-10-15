from pathlib import Path
import numpy as np

from config import (
    PROJECT_DIR,
    SEQUENCE_LENGTH,
    NUM_LANDMARKS,
    FEATURES_PER_FRAME,
    CAMERA_INDEX,
    ROI,
)

# ---------------------------------------------------------
# Word recognition configuration
# ---------------------------------------------------------

WORD_DATA_PATH = PROJECT_DIR / "WORD_Data"
WORD_MODEL_PATH = PROJECT_DIR / "models" / "word_lstm.keras"
WORD_LABELS_PATH = PROJECT_DIR / "models" / "word_labels.npy"

# Start with a small vocabulary. Add more words as you collect data.
WORDS = np.array([
    "HELLO",
    "BYE",
    "YOU",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "YES",
    "NO",
    "HELP",
], dtype="<U30")

WORD_NUM_SEQUENCES = 100
WORD_CONFIDENCE_THRESHOLD = 0.70
WORD_STABLE_FRAMES = 8

for p in (WORD_DATA_PATH, WORD_MODEL_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)

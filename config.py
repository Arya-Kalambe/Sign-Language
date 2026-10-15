from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "MP_Data"
MODEL_PATH = PROJECT_DIR / "models" / "asl_lstm.keras"
LABELS_PATH = PROJECT_DIR / "models" / "labels.npy"
HAND_MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

ACTIONS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
SEQUENCE_LENGTH = 30
NUM_SEQUENCES = 60
NUM_LANDMARKS = 21
FEATURES_PER_FRAME = NUM_LANDMARKS * 3
CAMERA_INDEX = 0
ROI = (40, 400, 0, 400)  # y1,y2,x1,x2
CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 8

for p in (DATA_PATH, MODEL_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)

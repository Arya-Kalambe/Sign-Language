from pathlib import Path
from urllib.request import urlopen
from config import HAND_MODEL_PATH

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 100000:
    print(f"Already exists: {HAND_MODEL_PATH}")
else:
    print("Downloading MediaPipe Hand Landmarker model...")
    with urlopen(URL, timeout=60) as r, open(HAND_MODEL_PATH, "wb") as f:
        f.write(r.read())
    print(f"Saved: {HAND_MODEL_PATH}")

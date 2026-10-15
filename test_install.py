import sys
print("Python:",sys.version)
import numpy as np; print("NumPy:",np.__version__)
import cv2; print("OpenCV:",cv2.__version__)
import mediapipe as mp; print("MediaPipe:",getattr(mp,"__version__","unknown"))
from mediapipe.tasks.python import vision; print("MediaPipe Tasks: OK")
import torch; print("PyTorch:",torch.__version__)
import keras; print("Keras:",keras.__version__,"backend:",keras.backend.backend())
print("All core imports succeeded.")

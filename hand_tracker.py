import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from config import HAND_MODEL_PATH, NUM_LANDMARKS

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

class HandTracker:
    def __init__(self, model_path=None, num_hands=1, min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5):
        model_path = str(model_path or HAND_MODEL_PATH)
        if not __import__('os').path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_path}. Run: python download_model.py")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)

    @staticmethod
    def extract_keypoints(result):
        if not result.hand_landmarks:
            return np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
        # Use the first detected hand; this project is single-hand ASL.
        hand = result.hand_landmarks[0]
        return np.asarray([[p.x, p.y, p.z] for p in hand], dtype=np.float32).reshape(-1)

    @staticmethod
    def draw(frame, result, offset=(0,0), scale=(1.0,1.0)):
        if not result.hand_landmarks:
            return frame
        ox, oy = offset
        sx, sy = scale
        for hand in result.hand_landmarks[:1]:
            pts=[]
            for p in hand:
                x=int(ox+p.x*sx); y=int(oy+p.y*sy)
                pts.append((x,y))
                cv2.circle(frame,(x,y),4,(0,255,0),-1)
            for a,b in HAND_CONNECTIONS:
                cv2.line(frame,pts[a],pts[b],(255,180,0),2)
        return frame

    def close(self):
        self.detector.close()

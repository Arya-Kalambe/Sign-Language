import cv2
import numpy as np
from pathlib import Path
from config import ROI, ACTIONS

def crop_roi(frame):
    y1,y2,x1,x2=ROI
    return frame[y1:y2,x1:x2]

def draw_ui(frame, title="ASL"):
    y1,y2,x1,x2=ROI
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
    cv2.putText(frame,title,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"ESC: quit",(10,frame.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return frame

def centered_text(frame,text,y,scale=.7):
    (w,h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    x=max(10,(frame.shape[1]-w)//2)
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)

def safe_mkdir(path): Path(path).mkdir(parents=True,exist_ok=True)

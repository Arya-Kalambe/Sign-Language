# Real-Time ASL A-Z — Python 3.14

This version fixes the original tutorial's biggest data problem: the old `data.py` loaded one image and copied its landmarks into all 30 frames. This project records **30 real consecutive webcam frames per sequence**.

## Pipeline
Webcam → ROI → MediaPipe Tasks Hand Landmarker → 21 landmarks × XYZ → 30-frame sequence → Keras 3 + PyTorch backend LSTM → A-Z prediction.

## Install
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch
python download_model.py
python test_install.py
```

## Collect data
Run:
```powershell
python collect_sequences.py
```
Press a letter key A-Z to select it, then press the same lowercase letter again to record a sequence. Each sequence is 30 consecutive frames. Start with at least 50–60 sequences per letter, across different hand positions/backgrounds/light conditions.

Data is stored as:
```
MP_Data/A/0/0.npy ... 29.npy
MP_Data/A/1/0.npy ... 29.npy
...
MP_Data/Z/59/0.npy ... 29.npy
```
Each `.npy` file contains 63 values: 21 landmarks × (x,y,z).

## Train
```powershell
python train_model.py
```
The model is saved to `models/asl_lstm.keras`; labels to `models/labels.npy`.

## Run real time
```powershell
python app.py
```
Press `C` to clear the current prediction and Esc to quit.

## Important
This recognizes isolated alphabet signs. It is **not yet a complete natural-language sentence translator**. Sentence generation needs a second layer for word segmentation/temporal signs and a language model or grammar layer.

For ASL, note that some letters such as J and Z are dynamic. A static single-hand classifier can recognize them only if the training sequences capture the required movement; for best results, treat dynamic letters separately or use a motion-aware model.

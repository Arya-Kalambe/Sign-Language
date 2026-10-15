import cv2
import time
import numpy as np
import shutil
from pathlib import Path

from config import (
    ACTIONS,
    DATA_PATH,
    SEQUENCE_LENGTH,
    NUM_SEQUENCES,
    CAMERA_INDEX
)

from hand_tracker import HandTracker
from utils import crop_roi, draw_ui, centered_text


"""
ASL A-Z Dataset Collector

CONTROLS
--------
A-Z       : Select a letter
SPACE     : Record one 30-frame sequence
BACKSPACE : Reset/delete all sequences for selected letter
ESC       : Quit

IMPORTANT
---------
Each sequence contains SEQUENCE_LENGTH consecutive webcam frames.

Example:

MP_Data/
    A/
        0/
            0.npy
            1.npy
            ...
            29.npy

        1/
            0.npy
            ...
            29.npy

    B/
        0/
        1/
        ...

The program NEVER overwrites existing sequences.
"""


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WINDOW_NAME = "ASL Dataset Collector"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540


# ---------------------------------------------------------
# Count existing sequences
# ---------------------------------------------------------

def existing_count(action):
    """
    Returns the number of existing sequence folders
    for a particular action.
    """

    if action is None:
        return 0

    path = DATA_PATH / action

    if not path.exists():
        return 0

    return len([
        x for x in path.iterdir()
        if x.is_dir()
    ])


# ---------------------------------------------------------
# Create A-Z directories
# ---------------------------------------------------------

for action in ACTIONS:
    action_path = DATA_PATH / action
    action_path.mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. "
        "Check CAMERA_INDEX in config.py."
    )


# ---------------------------------------------------------
# MediaPipe Hand Tracker
# ---------------------------------------------------------

tracker = HandTracker()


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

current = None

message = (
    "Press A-Z to select a sign"
)

recording = False


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

try:

    while True:

        # -------------------------------------------------
        # Read webcam frame
        # -------------------------------------------------

        ok, frame = cap.read()

        if not ok:
            print("Could not read frame from webcam.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # -------------------------------------------------
        # Crop Region of Interest
        # -------------------------------------------------

        roi = crop_roi(frame)

        # -------------------------------------------------
        # Detect hand
        # -------------------------------------------------

        result = tracker.detect(roi)

        # -------------------------------------------------
        # Draw landmarks
        # -------------------------------------------------

        tracker.draw(
            frame,
            result,
            offset=(0, 40),
            scale=(400, 360)
        )

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        if current:

            count = existing_count(current)

            selected_text = (
                f"Selected: {current} | "
                f"Sequences: {count}/{NUM_SEQUENCES}"
            )

        else:

            selected_text = (
                "Selected: - | Sequences: -"
            )

        draw_ui(
            frame,
            selected_text
        )

        centered_text(
            frame,
            message,
            440,
            0.65
        )

        # -------------------------------------------------
        # Keyboard
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # =================================================
        # ESC → QUIT
        # =================================================

        if key == 27:

            print("\nExiting dataset collector...")
            break


        # =================================================
        # A-Z → SELECT LETTER
        # =================================================

        if ord('a') <= key <= ord('z'):

            current = chr(key).upper()

            count = existing_count(current)

            if count >= NUM_SEQUENCES:

                message = (
                    f"{current} already has "
                    f"{NUM_SEQUENCES} sequences. "
                    f"Press BACKSPACE to reset."
                )

            else:

                message = (
                    f"Selected {current}. "
                    f"Press SPACE to record "
                    f"sequence #{count + 1}."
                )

            # IMPORTANT:
            # Do NOT record here.
            #
            # The user must press SPACE.
            continue


        # =================================================
        # BACKSPACE → RESET CURRENT LETTER
        # =================================================

        if key == 8:

            if current is None:

                message = (
                    "Select a letter first."
                )

                continue


            action_path = DATA_PATH / current

            if action_path.exists():

                for item in action_path.iterdir():

                    if item.is_dir():

                        shutil.rmtree(item)

                    else:

                        item.unlink()


            message = (
                f"Reset {current}. "
                f"Press SPACE to record."
            )

            continue


        # =================================================
        # SPACE → RECORD SEQUENCE
        # =================================================

        if key == 32:

            if current is None:

                message = (
                    "Select a letter first."
                )

                continue


            sequence_id = existing_count(
                current
            )


            # -------------------------------------------------
            # Maximum sequences reached
            # -------------------------------------------------

            if sequence_id >= NUM_SEQUENCES:

                message = (
                    f"{current} already has "
                    f"{NUM_SEQUENCES} sequences."
                )

                continue


            # -------------------------------------------------
            # Prepare recording
            # -------------------------------------------------

            recording = True

            buffer = []

            message = (
                f"Recording {current} "
                f"#{sequence_id + 1}/{NUM_SEQUENCES}"
            )


            print(
                f"\nRecording {current} "
                f"sequence {sequence_id + 1}/"
                f"{NUM_SEQUENCES}"
            )


            # =================================================
            # Capture 30 consecutive frames
            # =================================================

            while (
                recording
                and len(buffer) < SEQUENCE_LENGTH
            ):

                ok, fr = cap.read()

                if not ok:

                    print(
                        "Could not read webcam frame."
                    )

                    break


                # Mirror
                fr = cv2.flip(fr, 1)


                # -------------------------------------------------
                # ROI
                # -------------------------------------------------

                r = crop_roi(fr)


                # -------------------------------------------------
                # MediaPipe
                # -------------------------------------------------

                res = tracker.detect(r)


                # -------------------------------------------------
                # Draw landmarks
                # -------------------------------------------------

                tracker.draw(
                    fr,
                    res,
                    offset=(0, 40),
                    scale=(400, 360)
                )


                # -------------------------------------------------
                # Extract landmarks
                # -------------------------------------------------

                keypoints = tracker.extract_keypoints(
                    res
                )

                buffer.append(
                    keypoints
                )


                # -------------------------------------------------
                # Recording UI
                # -------------------------------------------------

                draw_ui(
                    fr,
                    (
                        f"RECORDING {current} | "
                        f"Frame {len(buffer)}/"
                        f"{SEQUENCE_LENGTH}"
                    )
                )


                centered_text(
                    fr,
                    "Keep your hand inside the box",
                    440,
                    0.65
                )


                cv2.imshow(
                    WINDOW_NAME,
                    fr
                )


                # -------------------------------------------------
                # Allow ESC while recording
                # -------------------------------------------------

                recording_key = (
                    cv2.waitKey(1) & 0xFF
                )

                if recording_key == 27:

                    raise KeyboardInterrupt


            # =================================================
            # Save sequence
            # =================================================

            if len(buffer) == SEQUENCE_LENGTH:

                output_dir = (
                    DATA_PATH
                    / current
                    / str(sequence_id)
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )


                for i, vector in enumerate(buffer):

                    np.save(
                        output_dir / f"{i}.npy",
                        vector
                    )


                print(
                    f"Saved {current} "
                    f"sequence #{sequence_id + 1}"
                )


                message = (
                    f"Saved {current} "
                    f"#{sequence_id + 1}. "
                    f"Press SPACE for another."
                )


            else:

                message = (
                    "Recording cancelled."
                )


            recording = False


        # -------------------------------------------------
        # Show normal webcam
        # -------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


finally:

    cap.release()

    tracker.close()

    cv2.destroyAllWindows()

    print("\nDataset collector closed.")
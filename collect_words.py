import cv2
import numpy as np
import shutil

from word_config import (
    WORDS,
    WORD_DATA_PATH,
    WORD_NUM_SEQUENCES,
)

from config import (
    SEQUENCE_LENGTH,
    CAMERA_INDEX,
)

from hand_tracker import HandTracker
from utils import crop_roi


# ============================================================
# SETTINGS
# ============================================================

WINDOW_NAME = "ASL Word Dataset Collector"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Larger box for collecting words
WORD_ROI = (
    10,     # top
    530,    # bottom
    80,     # left
    600,    # right
)


# ============================================================
# CREATE DATASET DIRECTORIES
# ============================================================

WORD_DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

for word in WORDS:

    (
        WORD_DATA_PATH / word
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COUNT EXISTING SEQUENCES
# ============================================================

def existing_count(word):

    word_path = (
        WORD_DATA_PATH / word
    )

    if not word_path.exists():
        return 0

    return len([
        x
        for x in word_path.iterdir()
        if x.is_dir()
    ])


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

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


# ============================================================
# HAND TRACKER
# ============================================================

tracker = HandTracker()


# ============================================================
# CURRENT WORD
# ============================================================

selected = 0

message = (
    "Press N/P to change word | "
    "SPACE to record | "
    "ESC to quit"
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ok, frame = cap.read()


        if not ok:

            print(
                "ERROR: Could not read webcam frame."
            )

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # CURRENT WORD
        # ----------------------------------------------------

        word = str(
            WORDS[selected]
        )

        count = existing_count(
            word
        )


        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        top, bottom, left, right = WORD_ROI


        roi = frame[
            top:bottom,
            left:right
        ]


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        result = tracker.detect(
            roi
        )


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        tracker.draw(

            frame,

            result,

            offset=(
                left,
                top
            ),

            scale=(
                right - left,
                bottom - top
            )
        )


        # ----------------------------------------------------
        # DRAW LARGE BOX
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (left, top),

            (right, bottom),

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # TOP BAR
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (0, 0),

            (960, 75),

            (25, 25, 25),

            -1
        )


        # Word name
        cv2.putText(

            frame,

            f"WORD: {word}",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2,

            cv2.LINE_AA
        )


        # Sequence count
        cv2.putText(

            frame,

            (
                f"Sequences: "
                f"{count}/{WORD_NUM_SEQUENCES}"
            ),

            (20, 62),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (0, 430),

            (960, 485),

            (25, 25, 25),

            -1
        )


        cv2.putText(

            frame,

            "N = Next Word | P = Previous Word | "
            "SPACE = Record | R = Reset | ESC = Exit",

            (10, 463),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        cv2.putText(

            frame,

            message,

            (10, 515),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # SHOW WINDOW
        # ----------------------------------------------------

        cv2.imshow(

            WINDOW_NAME,

            frame
        )


        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(
            1
        ) & 0xFF


        # ====================================================
        # ESC
        # ====================================================

        if key == 27:

            print(
                "Exiting word collector..."
            )

            break


        # ====================================================
        # NEXT WORD
        # ====================================================

        if key == ord("n"):

            selected = (
                selected + 1
            ) % len(WORDS)

            word = str(
                WORDS[selected]
            )

            message = (
                f"Selected {word}. "
                "Press SPACE to record."
            )

            continue


        # ====================================================
        # PREVIOUS WORD
        # ====================================================

        if key == ord("p"):

            selected = (
                selected - 1
            ) % len(WORDS)

            word = str(
                WORDS[selected]
            )

            message = (
                f"Selected {word}. "
                "Press SPACE to record."
            )

            continue


        # ====================================================
        # RESET CURRENT WORD
        # ====================================================

        if key == ord("r"):

            word = str(
                WORDS[selected]
            )

            word_path = (
                WORD_DATA_PATH / word
            )


            if word_path.exists():

                for item in word_path.iterdir():

                    if item.is_dir():

                        shutil.rmtree(
                            item
                        )

                    else:

                        item.unlink()


            message = (
                f"{word} dataset reset."
            )

            print(
                f"Reset dataset for {word}"
            )

            continue


        # ====================================================
        # RECORD
        # ====================================================

        if key == 32:

            word = str(
                WORDS[selected]
            )

            sequence_id = (
                existing_count(word)
            )


            # Check limit
            if (
                sequence_id
                >= WORD_NUM_SEQUENCES
            ):

                message = (
                    f"{word} already has "
                    f"{WORD_NUM_SEQUENCES} sequences."
                )

                continue


            print()
            print(
                "================================"
            )
            print(
                f"Recording: {word}"
            )
            print(
                f"Sequence: "
                f"{sequence_id + 1}/"
                f"{WORD_NUM_SEQUENCES}"
            )
            print(
                "================================"
            )


            message = (
                f"GET READY: "
                f"{word}..."
            )

            # ------------------------------------------------
            # RECORD 30 FRAMES
            # ------------------------------------------------

            buffer = []


            for frame_number in range(
                SEQUENCE_LENGTH
            ):

                ok, frame = cap.read()


                if not ok:

                    print(
                        "ERROR: "
                        "Could not read frame."
                    )

                    break


                frame = cv2.flip(
                    frame,
                    1
                )


                # --------------------------------------------
                # ROI
                # --------------------------------------------

                top, bottom, left, right = (
                    WORD_ROI
                )


                roi = frame[
                    top:bottom,
                    left:right
                ]


                # --------------------------------------------
                # DETECT HAND
                # --------------------------------------------

                result = tracker.detect(
                    roi
                )


                # --------------------------------------------
                # DRAW LANDMARKS
                # --------------------------------------------

                tracker.draw(

                    frame,

                    result,

                    offset=(
                        left,
                        top
                    ),

                    scale=(
                        right - left,
                        bottom - top
                    )
                )


                # --------------------------------------------
                # ROI BOX
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (left, top),

                    (right, bottom),

                    (0, 255, 0),

                    2
                )


                # --------------------------------------------
                # EXTRACT LANDMARKS
                # --------------------------------------------

                keypoints = (
                    tracker.extract_keypoints(
                        result
                    )
                )


                buffer.append(
                    keypoints
                )


                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (0, 0),

                    (960, 70),

                    (25, 25, 25),

                    -1
                )


                cv2.putText(

                    frame,

                    f"RECORDING: {word}",

                    (20, 30),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.75,

                    (0, 255, 0),

                    2,

                    cv2.LINE_AA
                )


                cv2.putText(

                    frame,

                    (
                        f"Frame "
                        f"{frame_number + 1}/"
                        f"{SEQUENCE_LENGTH}"
                    ),

                    (20, 60),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    1,

                    cv2.LINE_AA
                )


                cv2.imshow(

                    WINDOW_NAME,

                    frame
                )


                key_pressed = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                # ESC while recording
                if key_pressed == 27:

                    raise KeyboardInterrupt


            # ------------------------------------------------
            # SAVE SEQUENCE
            # ------------------------------------------------

            if len(buffer) == SEQUENCE_LENGTH:

                seq_dir = (
                    WORD_DATA_PATH
                    / word
                    / str(sequence_id)
                )


                seq_dir.mkdir(
                    parents=True,
                    exist_ok=False
                )


                for i, keypoints in enumerate(
                    buffer
                ):

                    np.save(

                        seq_dir
                        / f"{i}.npy",

                        keypoints
                    )


                print(
                    f"Saved {word} "
                    f"sequence "
                    f"{sequence_id + 1}"
                )


                message = (
                    f"Saved {word} "
                    f"sequence "
                    f"{sequence_id + 1}. "
                    "Press SPACE again."
                )


            else:

                message = (
                    "Recording incomplete. "
                    "Try again."
                )


finally:

    cap.release()

    tracker.close()

    cv2.destroyAllWindows()

    print(
        "Word collector closed."
    )


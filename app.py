import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import keras

from collections import deque

from config import (
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    CAMERA_INDEX,
    ROI,
    CONFIDENCE_THRESHOLD,
    STABLE_FRAMES,
)

from word_config import (
    WORD_MODEL_PATH,
    WORD_LABELS_PATH,
    WORD_CONFIDENCE_THRESHOLD,
    WORD_STABLE_FRAMES,
)

from hand_tracker import HandTracker
from utils import crop_roi, draw_ui
from sentence_generator import generate_rule_based, generate_with_gemini


# ============================================================
# LOAD MODELS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Alphabet model not found. Run train_model.py first."
    )

if not WORD_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Word model not found. Run collect_words.py and "
        "train_word_model.py first."
    )


# Load A-Z model
alphabet_model = keras.models.load_model(MODEL_PATH)

# Load Word model
word_model = keras.models.load_model(WORD_MODEL_PATH)


# ============================================================
# LOAD LABELS
# ============================================================

if LABELS_PATH.exists():

    alphabet_actions = np.load(
        LABELS_PATH,
        allow_pickle=True,
    )

else:

    alphabet_actions = np.array(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )


if WORD_LABELS_PATH.exists():

    word_actions = np.load(
        WORD_LABELS_PATH,
        allow_pickle=True,
    )

else:

    raise FileNotFoundError(
        "word_labels.npy not found."
    )


print("Alphabet classes:")
print(alphabet_actions)

print("\nWord classes:")
print(word_actions)


# ============================================================
# HAND TRACKER
# ============================================================

tracker = HandTracker()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cv2.namedWindow("ASL Sign Language to Text", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Sign Language to Text", 1280, 720)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    960,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    540,
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam. "
        "Check CAMERA_INDEX in config.py."
    )


# ============================================================
# STATE
# ============================================================

# Start in Alphabet mode
mode = "ALPHABET"


# ============================================================
# ROI SETTINGS
# ============================================================

# Existing ROI used for A-Z recognition
ALPHABET_ROI = ROI


# Larger ROI for WORD recognition
#
# Format:
#
# (top, bottom, left, right)
#
# Camera resolution:
# 960 × 540
#
# This gives approximately:
# 520 × 520
#
WORD_ROI = (
    10,     # top
    530,    # bottom
    80,     # left
    600,    # right
)


# ============================================================
# SEQUENCE
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

alphabet_history = deque(
    maxlen=STABLE_FRAMES
)

word_history = deque(
    maxlen=WORD_STABLE_FRAMES
)


# ============================================================
# CURRENT PREDICTION
# ============================================================

last_label = "-"

last_conf = 0.0


# ============================================================
# TEXT BUFFER
# ============================================================

text_buffer = []

# Sentence-generation comparison outputs. These do not alter either vision model.
rule_sentence = ""
gemini_sentence = ""
generation_status = "Press ENTER to compare generators"


# ============================================================
# DUPLICATE PREDICTION CONTROL
# ============================================================

last_committed_label = None


# Number of frames before another prediction
cooldown = 0

# After accepting a sign, wait until the hand leaves the ROI before
# accepting another sign. This prevents a held gesture from being appended
# repeatedly and prevents transient predictions from entering the text.
awaiting_hand_release = False
no_hand_frames = 0
RELEASE_FRAMES = 15


# ============================================================
# HELPER:
# CLEAR PREDICTION STATE
# ============================================================

def clear_prediction_state():

    global last_label
    global last_conf
    global last_committed_label
    global cooldown
    global awaiting_hand_release
    global no_hand_frames

    sequence.clear()

    alphabet_history.clear()

    word_history.clear()

    last_label = "-"

    last_conf = 0.0

    last_committed_label = None

    cooldown = 0
    awaiting_hand_release = False
    no_hand_frames = 0


# ============================================================
# HELPER:
# ADD PREDICTION TO TEXT
# ============================================================

def commit_label(label):
    """Accept one stable label; caller then waits for hand release."""
    global last_committed_label
    global cooldown

    label = str(label).strip()
    if not label or cooldown > 0 or label == last_committed_label:
        return False

    text_buffer.append(label)
    last_committed_label = label
    cooldown = 12
    return True


# ============================================================
# HELPER:
# DRAW PREDICTION BARS
# ============================================================

def draw_bars(
    frame,
    probs,
    actions,
):

    top = 70


    # Get top 5 predictions
    top_indices = np.argsort(
        probs
    )[-5:][::-1]


    for i in top_indices:

        probability = float(
            probs[i]
        )


        # Width of confidence bar
        bar_width = int(
            probability * 300
        )


        # Draw bar
        cv2.rectangle(

            frame,

            (430, top),

            (
                430 + bar_width,
                top + 25
            ),

            (40, 180, 80),

            -1,
        )


        # Prediction text
        cv2.putText(

            frame,

            f"{actions[i]} "
            f"{probability * 100:.1f}%",

            (440, top + 19),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA,
        )


        top += 32


# ============================================================
# HELPER:
# DRAW TEXT BUFFER
# ============================================================

def draw_text_buffer(frame):
    """Draw accepted labels and sentence outputs in a dedicated bottom panel."""
    text = " ".join(token for token in text_buffer if token)
    h, w = frame.shape[:2]
    panel_top = int(h * 0.78)
    cv2.rectangle(frame, (0, panel_top), (w, h), (25, 25, 25), -1)
    cv2.putText(frame, f"Accepted signs: {text[-75:]}", (12, panel_top + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Rules: {rule_sentence[-75:]}", (12, panel_top + 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 230, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gemini: {gemini_sentence[-75:]}", (12, panel_top + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, generation_status[:115], (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)


# ============================================================
# HELPER:
# DRAW ROI
# ============================================================

def draw_roi(
    frame,
    roi,
):

    top, bottom, left, right = roi


    # Green rectangle
    cv2.rectangle(

        frame,

        (left, top),

        (right, bottom),

        (0, 255, 0),

        2,
    )


    # Label
    cv2.putText(

        frame,

        (
            "WORD AREA"
            if mode == "WORDS"
            else "ALPHABET AREA"
        ),

        (left, max(25, top - 10)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2,

        cv2.LINE_AA,
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
                "Could not read frame "
                "from webcam."
            )

            break


        # Mirror image
        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # SELECT ROI
        # ----------------------------------------------------

        if mode == "WORDS":

            current_roi = WORD_ROI

        else:

            current_roi = ALPHABET_ROI


        top, bottom, left, right = current_roi


        # ----------------------------------------------------
        # CROP ROI
        # ----------------------------------------------------

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
            ),
        )


        # ----------------------------------------------------
        # DRAW ROI BOX
        # ----------------------------------------------------

        draw_roi(
            frame,
            current_roi
        )


        # ----------------------------------------------------
        # EXTRACT HAND FEATURES
        # ----------------------------------------------------

        hand_present = bool(getattr(result, "hand_landmarks", []))
        vec = tracker.extract_keypoints(result)

        if hand_present:
            no_hand_frames = 0
            if not awaiting_hand_release:
                sequence.append(vec)
        else:
            no_hand_frames += 1
            # No hand means no sign is being presented: never classify the
            # model's prediction on all-zero landmarks as a real sign.
            sequence.clear()
            alphabet_history.clear()
            word_history.clear()
            if no_hand_frames >= RELEASE_FRAMES:
                awaiting_hand_release = False
                last_committed_label = None
                cooldown = 0


        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if cooldown > 0:

            cooldown -= 1


        # ====================================================
        # PREDICTION
        # ====================================================

        if hand_present and not awaiting_hand_release and len(sequence) == SEQUENCE_LENGTH:


            # Convert sequence to numpy
            input_sequence = np.expand_dims(

                np.asarray(
                    sequence
                ),

                axis=0
            )


            # =================================================
            # ALPHABET MODEL
            # =================================================

            if mode == "ALPHABET":

                probs = alphabet_model.predict(

                    input_sequence,

                    verbose=0,
                )[0]


                actions = alphabet_actions

                threshold = (
                    CONFIDENCE_THRESHOLD
                )

                stable_frames = (
                    STABLE_FRAMES
                )

                history = (
                    alphabet_history
                )


            # =================================================
            # WORD MODEL
            # =================================================

            else:

                probs = word_model.predict(

                    input_sequence,

                    verbose=0,
                )[0]


                actions = word_actions

                threshold = (
                    WORD_CONFIDENCE_THRESHOLD
                )

                stable_frames = (
                    WORD_STABLE_FRAMES
                )

                history = (
                    word_history
                )


            # ------------------------------------------------
            # BEST PREDICTION
            # ------------------------------------------------

            idx = int(
                np.argmax(probs)
            )


            conf = float(
                probs[idx]
            )


            # Add prediction to history
            history.append(
                idx
            )


            # ------------------------------------------------
            # CHECK STABILITY
            # ------------------------------------------------

            if (

                len(history)
                == stable_frames

                and

                len(set(history))
                == 1

                and

                conf >= threshold

            ):

                label = str(
                    actions[idx]
                )


                last_label = label

                last_conf = conf


                # Accept the stable label once, then wait for the hand to
                # leave the ROI before another label can be accepted.
                if commit_label(label):
                    awaiting_hand_release = True
                    sequence.clear()
                    alphabet_history.clear()
                    word_history.clear()


            # ------------------------------------------------
            # DRAW CONFIDENCE BARS
            # ------------------------------------------------

            draw_bars(

                frame,

                probs,

                actions
            )


        # ====================================================
        # TOP STATUS BAR
        # ====================================================

        # Header and controls use separate rows (the previous UI drew both
        # strings at y=30, making them overlap).
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"Mode: {mode} | Current: {last_label} | Confidence: {last_conf * 100:.1f}%",
            (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        if awaiting_hand_release:
            control_status = "Sign accepted - remove your hand from the box to sign again"
        else:
            control_status = "Ready for a sign"
        cv2.putText(frame, control_status, (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (120, 230, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, "A Alphabet | W Words | SPACE Separator | ENTER Compare | R Rules | G Gemini",
                    (12, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, "BACKSPACE Undo | C Clear | ESC Quit",
                    (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                    (235, 235, 235), 1, cv2.LINE_AA)


        # ====================================================
        # TEXT OUTPUT
        # ====================================================

        draw_text_buffer(
            frame
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        # Display a larger frontend without changing the camera/model input size.
        display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("ASL Sign Language to Text", display_frame)


        # ====================================================
        # KEYBOARD INPUT
        # ====================================================

        key = cv2.waitKey(
            1
        ) & 0xFF


        # ----------------------------------------------------
        # ESC
        # ----------------------------------------------------

        if key == 27:

            break


        # ----------------------------------------------------
        # SENTENCE GENERATION: compare both on the same labels
        # ----------------------------------------------------
        if key in (10, 13):  # ENTER
            raw_sequence = " ".join(text_buffer).strip()
            if not raw_sequence:
                generation_status = "No recognized labels yet. Sign letters/words first."
            else:
                rule_sentence = generate_rule_based(raw_sequence)
                generation_status = "Generating Gemini output..."
                cv2.putText(frame, "Calling Gemini; please wait...", (10, 410),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("ASL Sign Language to Text", frame)
                try:
                    gemini_sentence = generate_with_gemini(raw_sequence)
                    generation_status = "Comparison complete: ENTER compares both outputs."
                except Exception as exc:
                    gemini_sentence = "Unavailable: configure GEMINI_API_KEY or check connection/model."
                    generation_status = f"Gemini error: {str(exc)[:90]}"
            continue

        if key == ord("r"):
            raw_sequence = " ".join(text_buffer).strip()
            rule_sentence = generate_rule_based(raw_sequence) if raw_sequence else ""
            generation_status = "Rule-based baseline generated."
            continue

        if key == ord("g"):
            raw_sequence = " ".join(text_buffer).strip()
            if not raw_sequence:
                generation_status = "No recognized labels yet."
            else:
                try:
                    gemini_sentence = generate_with_gemini(raw_sequence)
                    generation_status = "Gemini output generated."
                except Exception as exc:
                    gemini_sentence = "Unavailable: configure GEMINI_API_KEY or check connection/model."
                    generation_status = f"Gemini error: {str(exc)[:90]}"
            continue

        # ----------------------------------------------------
        # ALPHABET MODE
        # ----------------------------------------------------

        if key == ord("a"):

            mode = "ALPHABET"

            clear_prediction_state()

            continue


        # ----------------------------------------------------
        # WORD MODE
        # ----------------------------------------------------

        if key == ord("w"):

            mode = "WORDS"

            clear_prediction_state()

            continue


        # ----------------------------------------------------
        # SPACE
        # ----------------------------------------------------

        if key == 32:

            if (

                text_buffer

                and

                text_buffer[-1] != ""

            ):

                text_buffer.append("")

            rule_sentence = ""
            gemini_sentence = ""
            generation_status = "Sequence changed; press ENTER to regenerate."
            clear_prediction_state()

            continue


        # ----------------------------------------------------
        # BACKSPACE
        # ----------------------------------------------------

        if key == 8:

            if text_buffer:
                text_buffer.pop()
            rule_sentence = ""
            gemini_sentence = ""
            generation_status = "Sequence changed; press ENTER to regenerate."
            clear_prediction_state()

            continue


        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if key == ord("c"):

            text_buffer.clear()
            rule_sentence = ""
            gemini_sentence = ""
            generation_status = "Cleared. Sign a sequence, then press ENTER."

            clear_prediction_state()

            continue


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()

    tracker.close()

    cv2.destroyAllWindows()


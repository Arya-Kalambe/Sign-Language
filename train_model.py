# ============================================================
# 1. SET KERAS BACKEND
# ============================================================

import os

# Tell Keras to use the PyTorch backend.
# This must be set before importing Keras.
os.environ.setdefault("KERAS_BACKEND", "torch")


# ============================================================
# 2. IMPORT REQUIRED LIBRARIES
# ============================================================

import json
import numpy as np
import keras

from keras import layers

# train_test_split is used to divide our dataset into
# training data and testing data.
from sklearn.model_selection import train_test_split

# Import project-specific settings from config.py
#
# ACTIONS:
#   List of sign language classes, e.g. ["A", "B", "C", ...]
#
# DATA_PATH:
#   Location where the extracted sequence data is stored.
#
# SEQUENCE_LENGTH:
#   Number of frames in one sequence.
#
# FEATURES_PER_FRAME:
#   Number of numerical features extracted from one frame.
#
# MODEL_PATH:
#   Location where the trained model will be saved.
#
# LABELS_PATH:
#   Location where the class labels will be saved.
from config import (
    ACTIONS,
    DATA_PATH,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
    MODEL_PATH,
    LABELS_PATH
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset():

    # X will contain the input sequences/features.
    #
    # Example:
    # X = [
    #     [frame1, frame2, ..., frame30],   -> sequence 1
    #     [frame1, frame2, ..., frame30],   -> sequence 2
    # ]
    X = []

    # y will contain the class/category for each sequence.
    #
    # Example:
    # A -> 0
    # B -> 1
    # C -> 2
    y = []

    # Loop through every sign/action defined in ACTIONS.
    #
    # enumerate gives:
    # label_idx = numerical class index
    # action    = actual class name
    #
    # Example:
    # 0, "A"
    # 1, "B"
    # 2, "C"
    for label_idx, action in enumerate(ACTIONS):

        # Create the path for the current action.
        #
        # Example:
        # DATA_PATH / "A"
        # DATA_PATH / "B"
        action_dir = DATA_PATH / action

        # If the folder doesn't exist, skip this class.
        if not action_dir.exists():
            continue

        # Each subdirectory represents one complete sequence.
        #
        # Example:
        # A/
        #   0/
        #      0.npy
        #      1.npy
        #      ...
        #   1/
        #      0.npy
        #      1.npy
        #      ...
        #
        # sorted() keeps the sequences in a consistent order.
        for seq_dir in sorted(
            [p for p in action_dir.iterdir() if p.is_dir()]
        ):

            # This list will store all frames belonging
            # to the current sequence.
            frames = []

            # Assume the sequence is valid initially.
            # If anything goes wrong, we change this to False.
            valid = True

            # Read every frame from 0 to SEQUENCE_LENGTH - 1.
            #
            # If SEQUENCE_LENGTH = 30:
            #
            # 0.npy
            # 1.npy
            # 2.npy
            # ...
            # 29.npy
            for i in range(SEQUENCE_LENGTH):

                # Construct the path to the current frame.
                p = seq_dir / f"{i}.npy"

                # If the frame does not exist,
                # mark this sequence as invalid.
                if not p.exists():
                    valid = False
                    break

                # Load the NumPy file.
                #
                # astype("float32") converts the data into
                # 32-bit floating point numbers.
                arr = np.load(p).astype("float32")

                # Make sure the frame contains exactly the
                # expected number of features.
                #
                # Example:
                # If FEATURES_PER_FRAME = 63,
                # arr must have shape (63,)
                if arr.shape != (FEATURES_PER_FRAME,):
                    valid = False
                    break

                # Add the current frame's features
                # to the current sequence.
                frames.append(arr)

            # If all frames were successfully loaded,
            # add this sequence to the dataset.
            if valid:

                # Add the complete sequence to X.
                X.append(frames)

                # Add the numerical class label to y.
                #
                # Example:
                # If action = "C" and label_idx = 2,
                # then y gets 2.
                y.append(label_idx)

    # If no valid sequences were found,
    # stop the program and display an error.
    if not X:
        raise RuntimeError(
            f"No sequences found in {DATA_PATH}. "
            "Run collect_sequences.py first."
        )

    # Convert Python lists into NumPy arrays.
    #
    # X shape:
    # (number_of_sequences, sequence_length, features_per_frame)
    #
    # y shape:
    # (number_of_sequences,)
    return (
        np.asarray(X, dtype="float32"),
        np.asarray(y, dtype="int64")
    )


# ============================================================
# 4. LOAD THE DATA
# ============================================================

X, y = load_dataset()

# Print the shape of the dataset.
#
# Example:
# Dataset: (93, 30, 63)
#
# Meaning:
# 93 sequences
# 30 frames per sequence
# 63 features per frame
print("Dataset:", X.shape)


# Find which classes actually exist in the dataset.
#
# set(y.tolist()) removes duplicate labels.
# sorted() sorts the numerical labels.
#
# Then ACTIONS[i] converts the numerical label
# back to its class name.
print(
    "Classes present:",
    [ACTIONS[i] for i in sorted(set(y.tolist()))]
)


# A classification model needs at least two classes.
#
# For example:
# A vs B -> possible
#
# Only A -> impossible to train a classifier
if len(set(y.tolist())) < 2:
    raise RuntimeError("Need at least 2 classes to train.")


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Divide the dataset into:
#
# 85% -> training data
# 15% -> testing data
#
# random_state=42 makes the split reproducible.
#
# stratify=y ensures that the class distribution is
# approximately maintained in both training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. CONVERT LABELS INTO ONE-HOT ENCODING
# ============================================================

# Neural networks commonly use one-hot encoded labels
# when using categorical_crossentropy.
#
# Example:
#
# A -> [1, 0, 0]
# B -> [0, 1, 0]
# C -> [0, 0, 1]

y_train_cat = keras.utils.to_categorical(
    y_train,
    num_classes=len(ACTIONS)
)

y_test_cat = keras.utils.to_categorical(
    y_test,
    num_classes=len(ACTIONS)
)


# ============================================================
# 7. BUILD THE LSTM MODEL
# ============================================================

# Sequential means that the layers are connected
# one after another.
model = keras.Sequential([

    # Input shape:
    #
    # (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    #
    # Example:
    # (30, 63)
    #
    # This means:
    # 30 frames
    # 63 features per frame
    layers.Input(
        shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    ),


    # --------------------------------------------------------
    # FIRST LSTM LAYER
    # --------------------------------------------------------

    # LSTM is useful for sequence/time-series data.
    #
    # It processes the frames one by one and learns
    # temporal patterns in the sign movement.
    #
    # 64 = number of LSTM units.
    #
    # return_sequences=True means that this layer returns
    # an output for every frame.
    #
    # This is required because another LSTM layer comes next.
    layers.LSTM(
        64,
        return_sequences=True
    ),

    # Dropout randomly disables 20% of neurons during training.
    #
    # This helps reduce overfitting.
    layers.Dropout(0.2),


    # --------------------------------------------------------
    # SECOND LSTM LAYER
    # --------------------------------------------------------

    # The second LSTM receives the complete sequence
    # from the first LSTM.
    #
    # 128 units gives the model more capacity to learn
    # complex temporal patterns.
    #
    # return_sequences=True because another LSTM follows.
    layers.LSTM(
        128,
        return_sequences=True
    ),

    # Again, use dropout to reduce overfitting.
    layers.Dropout(0.2),


    # --------------------------------------------------------
    # THIRD LSTM LAYER
    # --------------------------------------------------------

    # The third LSTM has 64 units.
    #
    # return_sequences is not specified, so it defaults
    # to False.
    #
    # Therefore, this layer outputs one final representation
    # of the entire sequence.
    layers.LSTM(64),


    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    # Dense layer learns higher-level patterns from the
    # representation produced by the LSTM.
    #
    # 64 neurons with ReLU activation.
    layers.Dense(
        64,
        activation="relu"
    ),

    # Dropout to reduce overfitting.
    layers.Dropout(0.2),


    # Another fully connected layer.
    layers.Dense(
        32,
        activation="relu"
    ),


    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------

    # Number of neurons = number of sign language classes.
    #
    # Example:
    # If ACTIONS = ["A", "B", "C", "D"]
    # then this layer has 4 neurons.
    #
    # Softmax converts the outputs into probabilities.
    #
    # Example:
    # A -> 0.05
    # B -> 0.80
    # C -> 0.10
    # D -> 0.05
    #
    # The class with the highest probability is predicted.
    layers.Dense(
        len(ACTIONS),
        activation="softmax"
    )
])


# ============================================================
# 8. COMPILE THE MODEL
# ============================================================

model.compile(

    # Adam is an optimization algorithm used to update
    # the model's weights during training.
    #
    # 0.001 is the learning rate.
    optimizer=keras.optimizers.Adam(1e-3),

    # categorical_crossentropy is appropriate because
    # our labels are one-hot encoded.
    loss="categorical_crossentropy",

    # Accuracy tells us how many predictions are correct.
    metrics=["accuracy"]
)


# Display the model architecture,
# including layers, output shapes and number of parameters.
model.summary()


# ============================================================
# 9. DEFINE CALLBACKS
# ============================================================

callbacks = [

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    # Stop training if validation accuracy doesn't improve
    # for 12 consecutive epochs.
    #
    # restore_best_weights=True means that after stopping,
    # the model goes back to the weights from the epoch
    # that achieved the best validation accuracy.
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True
    ),


    # --------------------------------------------------------
    # MODEL CHECKPOINT
    # --------------------------------------------------------

    # Save the model whenever validation accuracy improves.
    #
    # MODEL_PATH comes from config.py.
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

model.fit(

    # Training input sequences
    X_train,

    # Training labels
    y_train_cat,

    # Data used to evaluate the model after every epoch
    validation_data=(X_test, y_test_cat),

    # Maximum number of times the model can go through
    # the entire training dataset.
    epochs=100,

    # Number of sequences processed at once.
    #
    # Example:
    # batch_size=16 means the model processes
    # 16 sequences before updating its weights.
    batch_size=16,

    # Use the callbacks defined above.
    callbacks=callbacks,

    # Shuffle training sequences before each epoch.
    # This generally helps the model learn better.
    shuffle=True
)


# ============================================================
# 11. EVALUATE THE TRAINED MODEL
# ============================================================

# Evaluate the model on the test dataset.
#
# This gives:
# loss     -> classification loss
# accuracy -> percentage of correct predictions
loss, acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)


# Convert accuracy from decimal to percentage.
#
# Example:
# 0.92 -> 92.00%
print(
    f"Test accuracy: {acc * 100:.2f}%"
)


# ============================================================
# 12. SAVE THE CLASS LABELS
# ============================================================

# Save ACTIONS so that during prediction we can convert
# a numerical prediction back to the actual sign.
#
# Example:
#
# Model predicts:
# 2
#
# labels:
# ["A", "B", "C", "D"]
#
# Therefore:
# 2 -> "C"
np.save(
    LABELS_PATH,
    ACTIONS
)


# ============================================================
# 13. PRINT SAVED FILE LOCATIONS
# ============================================================

print(
    f"Saved model: {MODEL_PATH}"
)

print(
    f"Saved labels: {LABELS_PATH}"
)
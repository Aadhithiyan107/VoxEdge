import pathlib
import numpy as np
import tensorflow as tf

from model import build_ds_cnn


# =========================
# Configuration
# =========================

DATA_PATH = pathlib.Path(
    r"D:\VoxEdge\data\processed"
)

MODEL_DIR = pathlib.Path(
    r"D:\VoxEdge\models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 64
EPOCHS = 30

CLASS_NAMES = [
    "down",
    "go",
    "left",
    "no",
    "right",
    "stop",
    "up",
    "yes"
]


# =========================
# Load processed dataset
# =========================

print("Loading processed dataset...")

X_train = np.load(
    DATA_PATH / "X_train.npy"
)

y_train = np.load(
    DATA_PATH / "y_train.npy"
)

X_val = np.load(
    DATA_PATH / "X_val.npy"
)

y_val = np.load(
    DATA_PATH / "y_val.npy"
)

X_test = np.load(
    DATA_PATH / "X_test.npy"
)

y_test = np.load(
    DATA_PATH / "y_test.npy"
)


print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# =========================
# Build model
# =========================

print("\nBuilding DS-CNN...")

model = build_ds_cnn(
    input_shape=X_train.shape[1:],
    num_classes=len(CLASS_NAMES)
)


# =========================
# Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# Callbacks
# =========================

checkpoint_path = (
    MODEL_DIR / "voxedge_ds_cnn_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# =========================
# Train
# =========================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)


# =========================
# Final test evaluation
# =========================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("FINAL BASELINE TEST RESULT")
print("==============================")

print(
    f"Test loss     : {test_loss:.4f}"
)

print(
    f"Test accuracy : {test_accuracy:.4f}"
)

print(
    f"Test accuracy : {test_accuracy * 100:.2f}%"
)


# =========================
# Save final model
# =========================

final_model_path = (
    MODEL_DIR / "voxedge_ds_cnn_final.keras"
)

model.save(
    final_model_path
)

print("\nModel saved to:")
print(final_model_path)
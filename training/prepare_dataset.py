import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

from audio_feature import extract_mfcc


# =========================
# Configuration
# =========================

DATASET_PATH = pathlib.Path(
    r"D:\VoxEdge\data\downloads"
    r"\mini_speech_commands_extracted"
    r"\mini_speech_commands"
)

CLASSES = [
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
# Load dataset
# =========================

X = []
y = []

print("Preparing dataset...\n")

for label, class_name in enumerate(CLASSES):

    class_path = DATASET_PATH / class_name
    wav_files = list(class_path.glob("*.wav"))

    print(
        f"{class_name:8s}: "
        f"{len(wav_files)} files"
    )

    for wav_file in wav_files:

        try:
            features = extract_mfcc(wav_file)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(
                f"Skipping {wav_file.name}: {e}"
            )


# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)


print("\nDataset loaded successfully!")

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# Add channel dimension
# =========================

X = X[..., np.newaxis]

print("X shape after channel:", X.shape)


# =========================
# Train / validation / test
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training   :", X_train.shape, y_train.shape)
print("Validation :", X_val.shape, y_val.shape)
print("Testing    :", X_test.shape, y_test.shape)


# =========================
# Save dataset
# =========================

OUTPUT_PATH = pathlib.Path(
    r"D:\VoxEdge\data\processed"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

np.save(OUTPUT_PATH / "X_train.npy", X_train)
np.save(OUTPUT_PATH / "y_train.npy", y_train)

np.save(OUTPUT_PATH / "X_val.npy", X_val)
np.save(OUTPUT_PATH / "y_val.npy", y_val)

np.save(OUTPUT_PATH / "X_test.npy", X_test)
np.save(OUTPUT_PATH / "y_test.npy", y_test)


print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)

print("\nDone!")
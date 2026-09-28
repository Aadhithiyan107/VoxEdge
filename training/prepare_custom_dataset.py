import pathlib
import random
import shutil

# ============================================================
# PATHS
# ============================================================

BASE_DIR = pathlib.Path(r"D:\VoxEdge\data")

CUSTOM_DIR = BASE_DIR / "custom_kws"

MINI_DIR = (
    BASE_DIR
    / "downloads"
    / "mini_speech_commands_extracted"
    / "mini_speech_commands"
)

OUTPUT_DIR = BASE_DIR / "custom_processed"

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)

# Positive keyword classes
KEYWORDS = [
    "voxedge",
    "jarvis",
    "friday",
    "edith",
    "karen",
]

# Number of unknown samples per Mini Speech Commands class
UNKNOWN_PER_CLASS = 50

# Number of noise/silence samples
NOISE_COUNT = 400

# Mini Speech Commands classes
UNKNOWN_CLASSES = [
    "down",
    "go",
    "left",
    "no",
    "right",
    "stop",
    "up",
    "yes",
]

# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

if OUTPUT_DIR.exists():
    print("Removing previous custom dataset...")
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_name in KEYWORDS + ["unknown", "noise"]:
    (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

# ============================================================
# COPY POSITIVE KEYWORDS
# ============================================================

print("\n==========================================")
print("COPYING POSITIVE KEYWORD DATA")
print("==========================================")

total_positive = 0

# VoxEdge: user's 100 samples
positive_sources = {
    "voxedge": [
        CUSTOM_DIR / "positive",
        CUSTOM_DIR / "voxedge",
    ],
    "jarvis": [
        CUSTOM_DIR / "jarvis",
    ],
    "friday": [
        CUSTOM_DIR / "friday",
    ],
    "edith": [
        CUSTOM_DIR / "edith",
    ],
    "karen": [
        CUSTOM_DIR / "karen",
    ],
}

for keyword, source_dirs in positive_sources.items():

    output_dir = OUTPUT_DIR / keyword
    files = []

    for source_dir in source_dirs:
        if source_dir.exists():
            files.extend(source_dir.glob("*.wav"))

    files = sorted(files)

    print(f"{keyword:8s}: {len(files)} files")

    for index, source_file in enumerate(files, start=1):

        destination = output_dir / f"{keyword}_{index:04d}.wav"

        shutil.copy2(
            source_file,
            destination
        )

    total_positive += len(files)

print("\nTotal positive samples:", total_positive)

# ============================================================
# CREATE UNKNOWN CLASS
# ============================================================

print("\n==========================================")
print("BUILDING UNKNOWN CLASS")
print("==========================================")

unknown_output = OUTPUT_DIR / "unknown"

total_unknown = 0

for class_name in UNKNOWN_CLASSES:

    class_dir = MINI_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Mini Speech Commands class not found: {class_dir}"
        )

    files = list(class_dir.glob("*.wav"))

    random.shuffle(files)

    selected = files[:UNKNOWN_PER_CLASS]

    print(
        f"{class_name:8s}: "
        f"{len(selected)} samples"
    )

    for index, source_file in enumerate(selected):

        destination = (
            unknown_output
            / f"unknown_{class_name}_{index:03d}.wav"
        )

        shutil.copy2(
            source_file,
            destination
        )

        total_unknown += 1

print("\nTotal unknown samples:", total_unknown)

# ============================================================
# BUILD NOISE / SILENCE CLASS
# ============================================================

print("\n==========================================")
print("BUILDING NOISE / SILENCE CLASS")
print("==========================================")

noise_output = OUTPUT_DIR / "noise"

# We will initially use short portions of Mini Speech Commands
# as background-like audio and create synthetic low-level noise.
#
# This is ONLY a first training baseline.
# Later we can replace/expand this with real environmental
# recordings from fan, keyboard, traffic, room noise, etc.

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000

noise_files = []

for class_name in UNKNOWN_CLASSES:

    class_dir = MINI_DIR / class_name

    if class_dir.exists():
        noise_files.extend(class_dir.glob("*.wav"))

random.shuffle(noise_files)

for index in range(NOISE_COUNT):

    # Every 5th sample = silence
    if index % 5 == 0:

        audio = np.zeros(
            NUM_SAMPLES,
            dtype=np.float32
        )

    else:

        # Generate low-level synthetic background noise
        audio = np.random.normal(
            0,
            0.003,
            NUM_SAMPLES
        ).astype(np.float32)

    output_file = (
        noise_output
        / f"noise_{index + 1:04d}.wav"
    )

    sf.write(
        output_file,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

total_noise = len(list(noise_output.glob("*.wav")))

print("Noise/silence samples:", total_noise)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("CUSTOM DATASET READY")
print("==========================================")

for class_name in KEYWORDS + ["unknown", "noise"]:

    count = len(
        list(
            (OUTPUT_DIR / class_name).glob("*.wav")
        )
    )

    print(
        f"{class_name:8s}: {count:4d} samples"
    )

print("\nDataset location:")
print(OUTPUT_DIR)

print("\nDone!")
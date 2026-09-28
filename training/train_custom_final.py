"""
VoxEdge Multi-Keyword DS-CNN Classifier Training Script
Trains a 6-class wake-word / keyword classifier using DS-CNN architecture:
Classes:
0 = voxedge
1 = jarvis
2 = friday
3 = edith
4 = karen
5 = unknown (includes background noise & non-keyword audio)
"""
import os
import random
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers, models

# ============================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================

ROOT = r"D:\VoxEdge"
DATASET_DIR = os.path.join(ROOT, "data", "custom_processed")
BASE_MODEL = os.path.join(ROOT, "models", "voxedge_ds_cnn_best.keras")
OUTPUT_BEST = os.path.join(ROOT, "models", "voxedge_multikeyword_best.keras")
OUTPUT_FINAL = os.path.join(ROOT, "models", "voxedge_multikeyword_final.keras")

SAMPLE_RATE = 16000
N_SAMPLES = 16000

N_MFCC = 10
N_FFT = 480
HOP_LENGTH = 320

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_MAP = {
    "voxedge": 0,
    "jarvis": 1,
    "friday": 2,
    "edith": 3,
    "karen": 4,
    "unknown": 5,
    "noise": 5   # Merge noise into unknown class (class 5)
}

CLASS_NAMES = ["voxedge", "jarvis", "friday", "edith", "karen", "unknown"]
NUM_CLASSES = len(CLASS_NAMES)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

# ============================================================
# DATASET LOADING & INSPECTION
# ============================================================

def get_audio_files(folder):
    files = []
    if not os.path.exists(folder):
        return files
    for root_dir, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root_dir, filename))
    return sorted(files)


def load_dataset_metadata():
    print("=" * 65)
    print("VOXEDGE MULTI-KEYWORD DATASET INSPECTION")
    print("=" * 65)

    all_paths = []
    all_labels = []
    class_file_counts = {name: 0 for name in ["voxedge", "jarvis", "friday", "edith", "karen", "unknown", "noise"]}

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    for sub_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, sub_folder)
        if os.path.isdir(folder_path):
            files = get_audio_files(folder_path)
            folder_lower = sub_folder.lower()
            if folder_lower in CLASS_MAP:
                target_cls = CLASS_MAP[folder_lower]
                class_file_counts[folder_lower] = len(files)
                for f in files:
                    all_paths.append(f)
                    all_labels.append(target_cls)

    print("\nExact file counts per directory:")
    for k, v in class_file_counts.items():
        print(f"  {k:12s}: {v}")
    print(f"Total raw audio files: {len(all_paths)}")

    # Class distribution mapping:
    merged_counts = {c_name: 0 for c_name in CLASS_NAMES}
    for label in all_labels:
        merged_counts[CLASS_NAMES[label]] += 1

    print("\nMerged 6-Class Distribution:")
    for c_name, count in merged_counts.items():
        print(f"  Class {CLASS_MAP[c_name]} ({c_name:10s}): {count} samples")

    return np.array(all_paths), np.array(all_labels)

# ============================================================
# FEATURE EXTRACTION & AUGMENTATION
# ============================================================

def extract_mfcc(audio):
    """MFCC extraction matching exact VoxEdge pipeline (10, 51, 1)."""
    # 1. Exact 1 second length
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    # 2. RMS Normalization
    rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    if rms > 1e-8:
        audio = audio * (0.08 / rms)

    # 3. MFCC Feature Extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # 4. Standard Score Normalization
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.astype(np.float32)


def augment_audio(audio):
    """Lightweight realistic audio augmentation for training only."""
    # 1. Gain variation (0.8x to 1.2x)
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        audio = audio * gain

    # 2. Small time shift (up to +/- 80ms = +/- 1280 samples)
    if random.random() < 0.5:
        shift = random.randint(-1280, 1280)
        audio = np.roll(audio, shift)

    # 3. Small background noise addition
    if random.random() < 0.4:
        noise_amp = random.uniform(0.001, 0.004)
        audio = audio + noise_amp * np.random.randn(len(audio)).astype(np.float32)

    return audio


def load_and_preprocess(path, is_training=False):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if is_training:
        audio = augment_audio(audio)
    mfcc = extract_mfcc(audio)
    return mfcc[..., np.newaxis]

# ============================================================
# DATASET GENERATOR / BATCH BUILDER
# ============================================================

def build_feature_dataset(paths, labels, is_training=False, num_augment_copies=1):
    X = []
    y = []
    total = len(paths)
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            # Original copy
            mfcc = load_and_preprocess(path, is_training=False)
            X.append(mfcc)
            y.append(label)

            # Additional augmented copies during training for minority classes if requested
            if is_training and num_augment_copies > 1:
                for _ in range(num_augment_copies - 1):
                    aug_mfcc = load_and_preprocess(path, is_training=True)
                    X.append(aug_mfcc)
                    y.append(label)
        except Exception as e:
            print(f"\nSkipping error file {path}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"\rPreprocessing features... [{i+1}/{total}]", end="")
    print()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    paths, labels = load_dataset_metadata()

    # 1. Stratified Train / Val / Test Split (70% Train, 15% Val, 15% Test)
    print("\n" + "=" * 65)
    print("DETERMINISTIC STRATIFIED DATA SPLIT")
    print("=" * 65)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )

    print(f"Train samples      : {len(train_paths)}")
    print(f"Validation samples : {len(val_paths)}")
    print(f"Test samples       : {len(test_paths)} (100% UNTOUCHED)")

    print("\nClass distribution per split:")
    print(f"{'Class':12s} | {'Train':6s} | {'Val':6s} | {'Test':6s}")
    print("-" * 40)
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tr_c = np.sum(train_labels == c_idx)
        va_c = np.sum(val_labels == c_idx)
        te_c = np.sum(test_labels == c_idx)
        print(f"{c_name:12s} | {tr_c:6d} | {va_c:6d} | {te_c:6d}")

    # 2. Compute Class Weights to balance loss during training
    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_vals)}
    print("\nCalculated Training Class Weights:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        print(f"  Class {c_idx} ({c_name:10s}): {class_weight_dict[c_idx]:.4f}")

    # 3. Extract Features
    print("\n" + "=" * 65)
    print("EXTRACTING MFCC FEATURES (Input shape: 10, 51, 1)")
    print("=" * 65)

    print("\nExtracting Training Data (with offline minority class augmentation)...")
    # Balance training set: produce 3x augmented copies for minority keyword classes
    X_train_list, y_train_list = [], []
    for p, l in zip(train_paths, train_labels):
        mfcc = load_and_preprocess(p, is_training=False)
        X_train_list.append(mfcc)
        y_train_list.append(l)
        # Apply 2 extra augmented copies for keyword classes (0 to 4)
        if l in [0, 1, 2, 3, 4]:
            for _ in range(2):
                aug_mfcc = load_and_preprocess(p, is_training=True)
                X_train_list.append(aug_mfcc)
                y_train_list.append(l)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)

    print(f"\nFinal Augmented Training Set Shape: {X_train.shape}")

    print("Extracting Validation Data...")
    X_val, y_val = build_feature_dataset(val_paths, val_labels, is_training=False)

    print("Extracting Untouched Test Data...")
    X_test, y_test = build_feature_dataset(test_paths, test_labels, is_training=False)

    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    # 4. Load Pretrained General DS-CNN Baseline
    print("\n" + "=" * 65)
    print("LOADING PRETRAINED GENERAL DS-CNN BASELINE")
    print("=" * 65)
    print(f"Loading base model: {BASE_MODEL}")

    base_model = tf.keras.models.load_model(BASE_MODEL)
    print(f"Base model input shape : {base_model.input_shape}")
    print(f"Base model output shape: {base_model.output_shape}")

    # Extract feature representation before classification head
    # Layer 'global_average_pooling2d' outputs 96-dim embedding
    gap_layer = base_model.get_layer("global_average_pooling2d")
    feature_extractor = models.Model(inputs=base_model.input, outputs=gap_layer.output)

    # Freeze feature extractor initially for Phase 1
    for layer in feature_extractor.layers:
        layer.trainable = False

    # 5. Build New 6-Class Multi-Keyword Classification Head
    inputs = feature_extractor.input
    features = feature_extractor.output

    x = layers.Dense(64, activation="relu", name="multi_dense")(features)
    x = layers.Dropout(0.3, name="multi_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="multi_keyword_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VoxEdge_MultiKeyword_DS_CNN")

    print("\nMulti-Keyword Model Architecture Summary:")
    model.summary()

    # Calculate parameter counts & float32 model size
    total_params = model.count_params()
    float32_size_kb = (total_params * 4) / 1024
    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Approx Float32 Size  : {float32_size_kb:.2f} KB ({float32_size_kb / 1024:.2f} MB)")
    print(f"Inference Input Shape: {model.input_shape}")

    # ============================================================
    # PHASE 1: TRAIN CLASSIFICATION HEAD ONLY
    # ============================================================
    print("\n" + "=" * 65)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD (FEATURE EXTRACTOR FROZEN)")
    print("=" * 65)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_BEST,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history_p1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # ============================================================
    # PHASE 2: FINE-TUNE DS-CNN CONVOLUTIONAL LAYERS
    # ============================================================
    print("\n" + "=" * 65)
    print("PHASE 2: FINE-TUNING CONVOLUTIONAL LAYERS (BATCHNORM FROZEN)")
    print("=" * 65)

    # Unfreeze Conv2D and DepthwiseConv2D layers, keep BatchNorm layers frozen
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            layer.trainable = True
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Keep classification head trainable
    model.get_layer("multi_dense").trainable = True
    model.get_layer("multi_keyword_output").trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_BEST,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_p2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # Save final model
    model.save(OUTPUT_FINAL)
    print(f"\nFinal model saved to: {OUTPUT_FINAL}")

    # ============================================================
    # UNTOUCHED TEST SET EVALUATION
    # ============================================================
    print("\n" + "=" * 65)
    print("EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 65)

    # Load best model checkpoint for evaluation
    best_model = tf.keras.models.load_model(OUTPUT_BEST)

    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\nOverall Test Loss    : {test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")

    # Detailed Per-Class Metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    print("\nPer-Class Detailed Evaluation Metrics:")
    print(f"{'Class Index':12s} | {'Keyword':10s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    print("-" * 75)
    for c_idx in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c_idx]
        print(f"Class {c_idx:6d} | {c_name:10s} | {precision[c_idx]*100:9.2f}% | {recall[c_idx]*100:9.2f}% | {f1[c_idx]*100:9.2f}% | {support[c_idx]:8d}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))

    print("\nConfusion Matrix (Rows = Ground Truth, Columns = Prediction):")
    header = "          " + " ".join([f"{CLASS_NAMES[i][:6]:>8s}" for i in range(NUM_CLASSES)])
    print(header)
    print("-" * len(header))
    for i in range(NUM_CLASSES):
        row_str = f"{CLASS_NAMES[i]:9s} " + " ".join([f"{cm[i, j]:8d}" for j in range(NUM_CLASSES)])
        print(row_str)

    # Summary Report
    print("\n" + "=" * 65)
    print("VOXEDGE MULTI-KEYWORD KWS MODEL REPORT SUMMARY")
    print("=" * 65)
    print(f"1. Dataset Raw Total Samples  : {len(paths)}")
    print(f"2. Split Counts               : Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    print(f"3. Model Architecture         : DS-CNN (Input: 10, 51, 1 -> Output: 6 softmax)")
    print(f"4. Total Parameter Count      : {total_params:,} ({float32_size_kb:.2f} KB)")
    print(f"5. Best Model Checkpoint Path : {OUTPUT_BEST}")
    print(f"6. Final Model Checkpoint Path: {OUTPUT_FINAL}")
    print(f"7. Overall Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"8. Individual Keyword Recalls :")
    for c_idx in range(NUM_CLASSES):
        print(f"   - {CLASS_NAMES[c_idx]:10s} Recall: {recall[c_idx]*100:.2f}% (F1: {f1[c_idx]*100:.2f}%)")

    # Feasibility evaluation
    is_good = (test_acc >= 0.85) and all(recall[i] >= 0.70 for i in range(5))
    print(f"\n9. Proceed to Live Microphone Testing? {'YES' if is_good else 'NEEDS FURTHER TUNING / REPORTED AS IS'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
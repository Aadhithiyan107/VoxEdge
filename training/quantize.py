import pathlib
import numpy as np
import tensorflow as tf

# ==============================
# PATHS
# ==============================

MODEL_PATH = pathlib.Path(
    r"D:\VoxEdge\models\voxedge_ds_cnn_best.keras"
)

DATA_PATH = pathlib.Path(
    r"D:\VoxEdge\data\processed"
)

OUTPUT_DIR = pathlib.Path(
    r"D:\VoxEdge\models"
)

FLOAT32_PATH = OUTPUT_DIR / "voxedge_ds_cnn_float32.tflite"
INT8_PATH = OUTPUT_DIR / "voxedge_ds_cnn_int8.tflite"


# ==============================
# LOAD MODEL
# ==============================

print("Loading best trained model...")
print("Model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")
print("Parameters:", model.count_params())


# ==============================
# LOAD REPRESENTATIVE DATA
# ==============================

print("\nLoading representative dataset...")

X_train = np.load(DATA_PATH / "X_train.npy")

print("Training data shape:", X_train.shape)


def representative_dataset():
    # Use a subset for calibration
    num_samples = min(500, len(X_train))

    for i in range(num_samples):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]


# ==============================
# FLOAT32 TFLITE
# ==============================

print("\n==============================")
print("FLOAT32 TFLITE CONVERSION")
print("==============================")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

float32_model = converter.convert()

with open(FLOAT32_PATH, "wb") as f:
    f.write(float32_model)

print("Float32 model saved:")
print(FLOAT32_PATH)

print(
    "Float32 size:",
    len(float32_model),
    "bytes",
    f"({len(float32_model) / 1024:.2f} KB)"
)


# ==============================
# FULL INT8 QUANTIZATION
# ==============================

print("\n==============================")
print("INT8 QUANTIZATION")
print("==============================")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

# Force integer-only operators
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Integer input/output for MCU deployment
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting to full INT8...")

int8_model = converter.convert()

with open(INT8_PATH, "wb") as f:
    f.write(int8_model)

print("\nINT8 model saved:")
print(INT8_PATH)

print(
    "INT8 size:",
    len(int8_model),
    "bytes",
    f"({len(int8_model) / 1024:.2f} KB)"
)


# ==============================
# SIZE COMPARISON
# ==============================

float_size = len(float32_model)
int8_size = len(int8_model)

reduction = (1 - int8_size / float_size) * 100

print("\n==============================")
print("MODEL SIZE COMPARISON")
print("==============================")

print(f"Float32 : {float_size / 1024:.2f} KB")
print(f"INT8    : {int8_size / 1024:.2f} KB")
print(f"Reduction: {reduction:.2f}%")


# ==============================
# VERIFY INT8 MODEL
# ==============================

print("\n==============================")
print("INT8 MODEL VERIFICATION")
print("==============================")

interpreter = tf.lite.Interpreter(
    model_path=str(INT8_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print("Shape:", input_details[0]["shape"])
print("dtype:", input_details[0]["dtype"])
print("quantization:", input_details[0]["quantization"])

print("\nOutput:")
print("Shape:", output_details[0]["shape"])
print("dtype:", output_details[0]["dtype"])
print("quantization:", output_details[0]["quantization"])

print("\nINT8 verification complete!")

print("\n==============================")
print("QUANTIZATION COMPLETE")
print("==============================")
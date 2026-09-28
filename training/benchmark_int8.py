import pathlib
import time
import numpy as np
import tensorflow as tf

# ==============================
# PATHS
# ==============================

MODEL_PATH = pathlib.Path(
    r"D:\VoxEdge\models\voxedge_ds_cnn_int8.tflite"
)

DATA_PATH = pathlib.Path(
    r"D:\VoxEdge\data\processed"
)


# ==============================
# LOAD TEST DATA
# ==============================

print("Loading test data...")

X_test = np.load(DATA_PATH / "X_test.npy")

print("Test data:", X_test.shape)


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH),
    num_threads=1
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]

print("Input shape:", input_details["shape"])
print("Input dtype:", input_details["dtype"])


# ==============================
# PREPARE ONE SAMPLE
# ==============================

sample = X_test[0:1]

quantized_input = np.round(
    sample / input_scale + input_zero_point
).astype(np.int8)


# ==============================
# WARM-UP
# ==============================

print("\nWarming up...")

for _ in range(20):
    interpreter.set_tensor(
        input_details["index"],
        quantized_input
    )
    interpreter.invoke()


# ==============================
# BENCHMARK
# ==============================

NUM_RUNS = 500

print(f"\nRunning {NUM_RUNS} inference tests...")

times = []

for _ in range(NUM_RUNS):

    start = time.perf_counter()

    interpreter.set_tensor(
        input_details["index"],
        quantized_input
    )

    interpreter.invoke()

    end = time.perf_counter()

    times.append(
        (end - start) * 1000
    )


times = np.array(times)


# ==============================
# RESULTS
# ==============================

print("\n==============================")
print("INT8 INFERENCE BENCHMARK")
print("==============================")

print(f"Runs          : {NUM_RUNS}")
print(f"Average       : {times.mean():.3f} ms")
print(f"Median        : {np.median(times):.3f} ms")
print(f"Minimum       : {times.min():.3f} ms")
print(f"Maximum       : {times.max():.3f} ms")
print(f"P95           : {np.percentile(times, 95):.3f} ms")
print(f"P99           : {np.percentile(times, 99):.3f} ms")

print("\n==============================")
print("MODEL FILE SIZE")
print("==============================")

model_size = MODEL_PATH.stat().st_size

print(f"Model size: {model_size} bytes")
print(f"Model size: {model_size / 1024:.2f} KB")

print("\n==============================")
print("BENCHMARK COMPLETE")
print("==============================")
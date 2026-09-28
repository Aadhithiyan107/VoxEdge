import pathlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# PATHS
# ==============================

DATA_PATH = pathlib.Path(
    r"D:\VoxEdge\data\processed"
)

MODEL_PATH = pathlib.Path(
    r"D:\VoxEdge\models\voxedge_ds_cnn_int8.tflite"
)

CLASS_NAMES = [
    "down", "go", "left", "no",
    "right", "stop", "up", "yes"
]


# ==============================
# LOAD TEST DATA
# ==============================

print("Loading test dataset...")

X_test = np.load(DATA_PATH / "X_test.npy")
y_test = np.load(DATA_PATH / "y_test.npy")

print("Test data:", X_test.shape)
print("Labels   :", y_test.shape)


# ==============================
# LOAD INT8 MODEL
# ==============================

print("\nLoading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print("Input dtype :", input_details["dtype"])
print("Input scale :", input_scale)
print("Input zero  :", input_zero_point)

print("Output dtype:", output_details["dtype"])
print("Output scale:", output_scale)
print("Output zero :", output_zero_point)


# ==============================
# INT8 INFERENCE
# ==============================

predictions = []

print("\nRunning INT8 inference...")

for i in range(len(X_test)):

    sample = X_test[i:i+1]

    # Quantize float input -> INT8
    quantized_input = np.round(
        sample / input_scale + input_zero_point
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input
    )

    interpreter.invoke()

    # Get INT8 output
    quantized_output = interpreter.get_tensor(
        output_details["index"]
    )

    # Dequantize output
    output = (
        quantized_output.astype(np.float32)
        - output_zero_point
    ) * output_scale

    predicted_class = np.argmax(output, axis=1)[0]

    predictions.append(predicted_class)


predictions = np.array(predictions)


# ==============================
# ACCURACY
# ==============================

accuracy = np.mean(predictions == y_test)

print("\n==============================")
print("INT8 TEST RESULT")
print("==============================")

print(f"Correct predictions : {np.sum(predictions == y_test)}")
print(f"Total samples       : {len(y_test)}")
print(f"INT8 accuracy       : {accuracy * 100:.2f}%")


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print("Rows = Actual")
print("Columns = Predicted\n")

print("       " + " ".join(f"{c:>6}" for c in CLASS_NAMES))

for i, row in enumerate(cm):
    print(
        f"{CLASS_NAMES[i]:>6} "
        + " ".join(f"{value:6d}" for value in row)
    )


# ==============================
# SAVE RESULTS
# ==============================

RESULTS_PATH = pathlib.Path(
    r"D:\VoxEdge\evaluation"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

np.savetxt(
    RESULTS_PATH / "int8_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

with open(
    RESULTS_PATH / "int8_accuracy.txt",
    "w"
) as f:
    f.write(
        f"INT8 Test Accuracy: {accuracy * 100:.2f}%\n"
    )

print("\nResults saved to:")
print(RESULTS_PATH)

print("\n==============================")
print("INT8 EVALUATION COMPLETE")
print("==============================")
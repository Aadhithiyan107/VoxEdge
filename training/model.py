import tensorflow as tf
from tensorflow.keras import layers, models


NUM_CLASSES = 8


def depthwise_separable_block(
    x,
    filters,
    stride=1,
    dropout=0.0
):
    """
    Lightweight Depthwise Separable Convolution block.
    """

    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise convolution
    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x


def build_ds_cnn(
    input_shape=(10, 51, 1),
    num_classes=NUM_CLASSES
):
    """
    Build compact DS-CNN for keyword spotting.
    """

    inputs = layers.Input(
        shape=input_shape,
        name="mfcc_input"
    )

    # Initial convolution
    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # DS-CNN blocks
    x = depthwise_separable_block(
        x,
        filters=32
    )

    x = depthwise_separable_block(
        x,
        filters=64,
        stride=2
    )

    x = depthwise_separable_block(
        x,
        filters=64
    )

    x = depthwise_separable_block(
        x,
        filters=96,
        stride=2
    )

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Small classifier
    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="keyword_output"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="VoxEdge_DS_CNN"
    )

    return model


if __name__ == "__main__":

    model = build_ds_cnn()

    model.summary()

    print("\nTotal parameters:")
    print(model.count_params())
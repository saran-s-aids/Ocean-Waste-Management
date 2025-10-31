from __future__ import annotations
from typing import Optional
import tensorflow as tf

from . import config


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (config.IMG_SIZE[0], config.IMG_SIZE[1], 3),
    base: str = config.BASE_MODEL,
    learning_rate: float = config.LEARNING_RATE,
    fine_tune_at: Optional[int] = config.FINE_TUNE_AT,
) -> tf.keras.Model:
    """Build a transfer-learning classifier with optional fine-tuning."""

    if base.lower() == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
    elif base.lower() == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
    else:
        raise ValueError("Unsupported base model: {base}")

    backbone.trainable = False  # freeze by default

    inputs = tf.keras.Input(shape=input_shape)
    # Inputs are scaled to [0,1] in the data pipeline; map to [-1,1] for MobileNetV2
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"marine_{base}_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Optionally unfreeze from a certain layer index for fine-tuning
    if fine_tune_at is not None:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate * 0.1),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    return model

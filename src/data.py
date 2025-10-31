from pathlib import Path
from typing import Tuple
import tensorflow as tf

from . import config

AUTOTUNE = tf.data.AUTOTUNE


def get_datasets(
    data_dir: Path | str,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Training dataset with built-in split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        labels="inferred",
        class_names=config.CLASS_NAMES,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        labels="inferred",
        class_names=config.CLASS_NAMES,
    )

    class_names = list(config.CLASS_NAMES)

    # Drop any elements that error during decode
    try:
        # Prefer the modern API if available
        if hasattr(train_ds, "ignore_errors"):
            train_ds = train_ds.ignore_errors()
            val_ds = val_ds.ignore_errors()
        else:
            train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
            val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    except Exception:
        pass

    # Data augmentation + normalization pipeline
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")

    def augment_and_normalize(x, y):
        x = data_augmentation(x, training=True)
        x = normalization(x)
        return x, y

    def normalize_only(x, y):
        x = normalization(x)
        return x, y

    train_ds = train_ds.map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(normalize_only, num_parallel_calls=AUTOTUNE)

    # Performance optimizations
    train_ds = train_ds.shuffle(8 * batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names

from pathlib import Path
import tensorflow as tf

from . import config
from .data import get_datasets
from .model import build_model
from .utils import ensure_output_dirs, plot_training_curves


def train(
    data_dir: Path | str = config.DATA_DIR,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    epochs: int = config.EPOCHS,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
):
    ensure_output_dirs()

    train_ds, val_ds, class_names = get_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    model = build_model(num_classes=len(class_names), input_shape=(img_size[0], img_size[1], 3))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH), monitor="val_accuracy", save_best_only=True, save_weights_only=False
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training curves
    plot_training_curves(history, config.PLOTS_DIR / "training_curves.png")

    # Ensure best model is saved
    model.save(config.MODEL_PATH)

    return model, val_ds, class_names, history

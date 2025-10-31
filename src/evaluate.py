from pathlib import Path
import tensorflow as tf

from . import config
from .utils import ensure_output_dirs, evaluate_and_report


def evaluate(model_path: Path | str = config.MODEL_PATH, val_ds: tf.data.Dataset | None = None, class_names: list[str] | None = None):
    ensure_output_dirs()

    # Lazy import to avoid circular
    if val_ds is None or class_names is None:
        from .data import get_datasets
        _, val_ds, class_names = get_datasets(
            data_dir=config.DATA_DIR,
            img_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            seed=config.SEED,
        )

    model = tf.keras.models.load_model(model_path)

    cm_outfile = config.PLOTS_DIR / "confusion_matrix.png"
    report_outfile = config.OUTPUTS_DIR / "metrics_report.txt"

    summary = evaluate_and_report(model, val_ds, tuple(class_names), cm_outfile, report_outfile)

    print("\nEvaluation summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved confusion matrix: {cm_outfile}")
    print(f"Saved metrics report: {report_outfile}")

    return summary

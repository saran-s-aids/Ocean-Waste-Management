from pathlib import Path
from typing import Dict, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from . import config


def ensure_output_dirs():
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history: tf.keras.callbacks.History, outfile: Path):
    metrics = ["accuracy", "loss"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label=f"train_{metric}")
        ax.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


def evaluate_and_report(model: tf.keras.Model, ds: tf.data.Dataset, class_names: Tuple[str, ...], cm_outfile: Path, report_outfile: Path) -> Dict:
    y_true = []
    y_pred = []
    for batch_x, batch_y in ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_y.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_outfile, dpi=150)
    plt.close()

    cls_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    summary = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    with open(report_outfile, "w", encoding="utf-8") as f:
        f.write("Marine Plastic Classifier - Evaluation Report\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n\nClassification Report (per class)\n")
        f.write(cls_report)

    return summary

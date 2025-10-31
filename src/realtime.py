import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

from . import config


def realtime_demo(model_path: Path | str = config.MODEL_PATH, img_size: tuple[int, int] = config.IMG_SIZE, class_names: list[str] | None = None):
    model = tf.keras.models.load_model(model_path)

    # Try to infer class names from training config if not provided
    if class_names is None:
        class_names = config.CLASS_NAMES

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size)
        x = img_resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = f"{class_names[idx]}: {conf*100:.1f}%"

        # Draw label
        color = (0, 200, 0) if idx == 0 else (0, 165, 255) if idx == 1 else (50, 50, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.imshow("Marine Plastic Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

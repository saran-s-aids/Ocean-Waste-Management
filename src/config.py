from pathlib import Path

# Core config
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODEL_PATH = Path("marine_plastic_classifier.h5")

# Training hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
SEED = 42

# Model settings
BASE_MODEL = "mobilenetv2"  # options: mobilenetv2, efficientnetb0 (requires tf>=2.13)
LEARNING_RATE = 1e-3
FINE_TUNE_AT = None  # e.g., set to an integer layer index to partially fine-tune base model

# Classes (folders under DATA_DIR)
# Ensure your dataset has exactly these subfolders
CLASS_NAMES = ["plastic", "organic", "other"]

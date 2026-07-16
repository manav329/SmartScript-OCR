"""Centralized configuration for the MNIST digit classification project."""

from pathlib import Path


# Project root resolved from src/config.py -> project_root/src/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# Data and artifact directories
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODEL_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
IMAGES_TEST_DIR: Path = PROJECT_ROOT / "images" / "test"


DIRECTORY_PATHS: tuple[Path, ...] = (
	RAW_DATA_DIR,
	PROCESSED_DATA_DIR,
	MODEL_DIR,
	CHECKPOINT_DIR,
	LOGS_DIR,
	REPORTS_DIR,
	IMAGES_TEST_DIR,
)


# Image dimensions for MNIST samples
IMG_HEIGHT: int = 28
IMG_WIDTH: int = 28
IMG_CHANNELS: int = 1


# Classification settings
NUM_CLASSES: int = 10


# Training hyperparameters
BATCH_SIZE: int = 128
EPOCHS: int = 15
LEARNING_RATE: float = 0.001
VALIDATION_SPLIT: float = 0.1


# Reproducibility
RANDOM_SEED: int = 42

# Constant model name and path
MODEL_NAME = "mnist_cnn_model.v1"
MODEL_SAVE_PATH = MODEL_DIR / f"{MODEL_NAME}.keras"
FINAL_MODEL_PATH: Path = MODEL_DIR / "mnist_cnn_final.keras"


def ensure_directories() -> None:
	"""Create and verify required project directories.

	Ensures all configured data, model, checkpoint, logging, and reporting
	directories exist on disk.

	Returns:
		None
	"""
	for directory in DIRECTORY_PATHS:
		directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
	ensure_directories()
	print("Created/verified project directories:")
	for directory in DIRECTORY_PATHS:
		print(f"[OK] {directory}")

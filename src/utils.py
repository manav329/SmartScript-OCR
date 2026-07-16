"""Shared utilities for handwritten recognition."""

from __future__ import annotations

if __package__ is None or __package__ == "":
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.predict import load_trained_model, predict_digit


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def load_and_prepare_custom_image(
	image_path: str,
	invert_if_needed: bool = True,
) -> np.ndarray:
	"""Load an image and convert it to MNIST-style 28x28 grayscale pixels.

	MNIST digits are represented as white/bright strokes on a black/dark
	background. Typical real-world photos or scans of handwritten digits are often
	the opposite (dark ink on white paper). This helper optionally auto-inverts
	such images so custom inputs match the model's training convention.

	Args:
		image_path: Path to an image on disk.
		invert_if_needed: If ``True``, invert when the image is mostly light
			(mean pixel value > 127), which usually indicates dark foreground on
			light background.

	Returns:
		A ``(28, 28)`` ``np.uint8`` image array with MNIST-like polarity
		(white digit, black background).

	Raises:
		FileNotFoundError: If ``image_path`` does not exist.
		Exception: If PIL fails to open or process the image for other reasons.
	"""
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image file not found: {path}")

	try:
		with Image.open(path) as image:
			grayscale_image = image.convert("L")
			resized_image = grayscale_image.resize((28, 28), Image.LANCZOS)
			pixels = np.array(resized_image, dtype=np.uint8)
	except Exception as exc:
		raise Exception(f"Failed to load and prepare custom image '{path}': {exc}") from exc

	if invert_if_needed and float(np.mean(pixels)) > 127.0:
		pixels = 255 - pixels

	return pixels.astype(np.uint8)


if __name__ == "__main__":
	project_root = Path(__file__).resolve().parent.parent
	model_path = project_root / "models" / "mnist_cnn_final.keras"
	image_dir = project_root / "images" / "test"

	model = load_trained_model(str(model_path))

	if not image_dir.exists():
		raise FileNotFoundError(f"Test image directory not found: {image_dir}")

	image_paths = sorted(
		[path for path in image_dir.iterdir() if path.is_file() and not path.name.startswith(".")],
		key=lambda path: path.name.lower(),
	)

	if not image_paths:
		LOGGER.info("No images found in %s", image_dir)

	for path in image_paths:
		prepared_image = load_and_prepare_custom_image(str(path), invert_if_needed=True)
		predicted_digit, confidence = predict_digit(model, prepared_image)
		print(f"{path.name} -> Predicted: {predicted_digit}, Confidence: {confidence:.4f}")

"""Prediction utilities for handwritten recognition."""

from __future__ import annotations

if __package__ is None or __package__ == "":
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
from data_loader import load_mnist
from preprocess import normalize_images, reshape_for_cnn


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def load_trained_model(model_path: str) -> tf.keras.Model:
	"""Load a trained Keras model from disk.

	Args:
		model_path: Filesystem path to the saved ``.keras`` model file.

	Returns:
		The loaded ``tf.keras.Model`` instance.

	Raises:
		FileNotFoundError: If ``model_path`` does not exist.
		RuntimeError: If the model exists but cannot be loaded.
	"""
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Trained model file not found: {path}")

	try:
		return tf.keras.models.load_model(path)
	except Exception as exc:  # pragma: no cover - depends on runtime model state.
		raise RuntimeError(f"Failed to load trained model from {path}.") from exc


def preprocess_single_image(image: np.ndarray) -> np.ndarray:
	"""Preprocess a single grayscale image for CNN inference.

	The input must already be a 2D MNIST-sized grayscale image with shape
	``(IMG_HEIGHT, IMG_WIDTH)``. The same normalization and reshaping helpers
	used during training are applied to a batch of size 1 so inference stays
	identical to the training pipeline.
	"""
	if not isinstance(image, np.ndarray):
		raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}.")

	if image.shape != (IMG_HEIGHT, IMG_WIDTH):
		raise ValueError(
			"image must have shape "
			f"{(IMG_HEIGHT, IMG_WIDTH)}, got {tuple(image.shape)}."
		)

	image_batch = np.expand_dims(image, axis=0)
	return reshape_for_cnn(normalize_images(image_batch))


def predict_digit(model: tf.keras.Model, image: np.ndarray) -> tuple[int, float]:
	"""Predict a single handwritten digit and return its confidence score."""
	preprocessed_image = preprocess_single_image(image)
	predictions = model.predict(preprocessed_image, verbose=0)
	prediction = predictions[0]
	predicted_class = int(np.argmax(prediction))
	confidence = float(np.max(prediction))
	return predicted_class, confidence


def predict_batch(model: tf.keras.Model, images: np.ndarray) -> list[tuple[int, float]]:
	"""Predict handwritten digits for a batch of grayscale images.

	Args:
		model: Trained Keras model.
		images: Array with shape ``(N, IMG_HEIGHT, IMG_WIDTH)``.

	Returns:
		A list of ``(predicted_class, confidence)`` tuples in input order.
	"""
	if not isinstance(images, np.ndarray):
		raise TypeError(f"images must be a numpy.ndarray, got {type(images).__name__}.")

	if images.ndim != 3 or images.shape[1:] != (IMG_HEIGHT, IMG_WIDTH):
		raise ValueError(
			"images must have shape "
			f"(N, {IMG_HEIGHT}, {IMG_WIDTH}), got {tuple(images.shape)}."
		)

	preprocessed_images = reshape_for_cnn(normalize_images(images))
	predictions = model.predict(preprocessed_images, verbose=0)
	predicted_classes = np.argmax(predictions, axis=1)
	confidences = np.max(predictions, axis=1)
	return [
		(int(predicted_class), float(confidence))
		for predicted_class, confidence in zip(predicted_classes, confidences)
	]


def _print_prediction_table(
	indices: np.ndarray,
	true_labels: np.ndarray,
	predictions: list[tuple[int, float]],
) -> None:
	"""Print a compact table comparing true labels and predictions."""
	header = f"{'Index':>5} | {'True':>4} | {'Predicted':>9} | {'Confidence':>10} | {'Match':>5}"
	separator = "-" * len(header)
	print(header)
	print(separator)
	for index, true_label, (predicted_label, confidence) in zip(indices, true_labels, predictions):
		match = "YES" if int(true_label) == int(predicted_label) else "NO"
		print(
			f"{int(index):>5} | {int(true_label):>4} | {int(predicted_label):>9} | "
			f"{confidence:>10.4f} | {match:>5}"
		)


if __name__ == "__main__":
	model_path = Path(__file__).resolve().parent.parent / "models" / "mnist_cnn_final.keras"
	model = load_trained_model(str(model_path))

	(_, _), (x_test, y_test) = load_mnist()
	random_generator = np.random.default_rng()
	sample_count = min(5, len(x_test))
	sample_indices = random_generator.choice(len(x_test), size=sample_count, replace=False)
	sample_images = x_test[sample_indices]
	sample_labels = y_test[sample_indices]
	sample_predictions = predict_batch(model, sample_images)

	_print_prediction_table(sample_indices, sample_labels, sample_predictions)

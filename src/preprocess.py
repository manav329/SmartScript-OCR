"""Preprocessing utilities for handwritten recognition."""

from __future__ import annotations

if __package__ is None or __package__ == "":
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
	IMG_CHANNELS,
	IMG_HEIGHT,
	IMG_WIDTH,
	NUM_CLASSES,
	RANDOM_SEED,
	VALIDATION_SPLIT,
)


def _ensure_numpy_array(name: str, value: np.ndarray) -> np.ndarray:
	"""Validate that a value is a NumPy array."""
	if not isinstance(value, np.ndarray):
		raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}.")
	return value


def normalize_images(images: np.ndarray) -> np.ndarray:
	"""Cast image pixels to float32 and scale them to the [0, 1] range.

	This normalization matches the value range expected by Keras CNNs and helps
	improve numerical stability during training compared with raw uint8 pixels in
	the 0-255 range.
	"""
	_ensure_numpy_array("images", images)
	return images.astype(np.float32) / 255.0


def reshape_for_cnn(images: np.ndarray) -> np.ndarray:
	"""Reshape flat MNIST images into 4D tensors for a CNN.

	The returned array has shape ``(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)``.
	"""
	_ensure_numpy_array("images", images)

	if images.size % (IMG_HEIGHT * IMG_WIDTH) != 0:
		raise ValueError(
			"images size must be divisible by IMG_HEIGHT * IMG_WIDTH "
			f"({IMG_HEIGHT * IMG_WIDTH}), but got {images.size}."
		)

	return images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


def encode_labels(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
	"""One-hot encode integer class labels for categorical classification."""
	_ensure_numpy_array("labels", labels)
	return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


def split_train_validation(
	x_train: np.ndarray,
	y_train: np.ndarray,
	val_split: float = VALIDATION_SPLIT,
	seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Split training data into training and validation subsets.

	The split is stratified on the integer labels so each class stays represented
	proportionally in both subsets.
	"""
	_ensure_numpy_array("x_train", x_train)
	_ensure_numpy_array("y_train", y_train)

	if len(x_train) != len(y_train):
		raise ValueError(
			"x_train and y_train must contain the same number of samples: "
			f"got {len(x_train)} and {len(y_train)}."
		)

	x_train_split, x_val, y_train_split, y_val = train_test_split(
		x_train,
		y_train,
		test_size=val_split,
		random_state=seed,
		stratify=y_train,
	)
	return x_train_split, x_val, y_train_split, y_val


def preprocess_pipeline(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
) -> dict[str, np.ndarray]:
	"""Run the full MNIST preprocessing pipeline.

	The pipeline splits the training data first, then normalizes, reshapes, and
	one-hot encodes the labels for all splits.
	"""
	_ensure_numpy_array("x_train", x_train)
	_ensure_numpy_array("y_train", y_train)
	_ensure_numpy_array("x_test", x_test)
	_ensure_numpy_array("y_test", y_test)

	x_train_split, x_val, y_train_split, y_val = split_train_validation(x_train, y_train)

	x_train_split = reshape_for_cnn(normalize_images(x_train_split))
	x_val = reshape_for_cnn(normalize_images(x_val))
	x_test = reshape_for_cnn(normalize_images(x_test))

	y_train_split = encode_labels(y_train_split)
	y_val = encode_labels(y_val)
	y_test = encode_labels(y_test)

	return {
		"x_train": x_train_split,
		"y_train": y_train_split,
		"x_val": x_val,
		"y_val": y_val,
		"x_test": x_test,
		"y_test": y_test,
	}


def verify_preprocessing_output(processed: dict[str, np.ndarray]) -> None:
	"""Verify the preprocessing output against expected MNIST CNN shapes.

	This checks the validation split size, normalized pixel range, and that the
	first training label is a valid one-hot vector.
	"""
	required_keys = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}
	missing_keys = required_keys.difference(processed)
	if missing_keys:
		raise KeyError(f"processed is missing required keys: {sorted(missing_keys)}")

	x_train = processed["x_train"]
	y_train = processed["y_train"]
	x_val = processed["x_val"]
	x_test = processed["x_test"]

	expected_train_size = int(60000 * (1.0 - VALIDATION_SPLIT))
	expected_val_size = int(60000 * VALIDATION_SPLIT)

	if x_train.shape != (expected_train_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
		raise AssertionError(
			"Unexpected x_train shape: "
			f"{x_train.shape}, expected {(expected_train_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)}."
		)

	if x_val.shape != (expected_val_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
		raise AssertionError(
			"Unexpected x_val shape: "
			f"{x_val.shape}, expected {(expected_val_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)}."
		)

	if x_test.shape != (10000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
		raise AssertionError(
			"Unexpected x_test shape: "
			f"{x_test.shape}, expected {(10000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)}."
		)

	if not np.isclose(float(x_train.min()), 0.0):
		raise AssertionError(f"x_train.min() expected to be 0.0, got {float(x_train.min())}.")

	if not np.isclose(float(x_train.max()), 1.0):
		raise AssertionError(f"x_train.max() expected to be 1.0, got {float(x_train.max())}.")

	if not np.isclose(float(y_train[0].sum()), 1.0):
		raise AssertionError(f"y_train[0].sum() expected to be 1.0, got {float(y_train[0].sum())}.")

	print("Verification passed: expected shapes, normalized range, and one-hot encoding are correct.")


if __name__ == "__main__":
	from src.data_loader import load_mnist

	(x_train, y_train), (x_test, y_test) = load_mnist()
	processed = preprocess_pipeline(x_train, y_train, x_test, y_test)

	for name, array in processed.items():
		print(f"{name} shape {array.shape} dtype {array.dtype}")

	verify_preprocessing_output(processed)

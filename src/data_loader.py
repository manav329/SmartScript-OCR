"""Data loading utilities for the MNIST handwritten digit dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

_MNIST_CACHE_PATH = Path.home() / ".keras" / "datasets" / "mnist.npz"
_CACHE_LOCATION_LOGGED = False


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
	"""Load the raw MNIST dataset.

	Returns:
		A pair of ``(x_train, y_train)`` and ``(x_test, y_test)`` tuples. The
		returned arrays are the unmodified raw arrays provided by TensorFlow:
		``uint8`` image pixels and integer labels.

	Raises:
		RuntimeError: If the dataset cannot be downloaded or loaded.
	"""

	global _CACHE_LOCATION_LOGGED

	if not _CACHE_LOCATION_LOGGED:
		LOGGER.info("MNIST cache location: %s", _MNIST_CACHE_PATH)
		_CACHE_LOCATION_LOGGED = True

	try:
		return tf.keras.datasets.mnist.load_data()
	except Exception as exc:  # pragma: no cover - exercised via runtime failure.
		if _MNIST_CACHE_PATH.exists():
			LOGGER.warning(
				"MNIST cache file exists at %s but could not be loaded; the cache may be corrupted.",
				_MNIST_CACHE_PATH,
			)
		raise RuntimeError(
			"Failed to load the MNIST dataset via tf.keras.datasets.mnist.load_data(). "
			"Check your internet connection or remove a possibly corrupted local cache file and try again."
		) from exc


def describe_dataset(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
) -> None:
	"""Print and log a concise summary of the MNIST dataset.

	Args:
		x_train: Training images.
		y_train: Training labels.
		x_test: Test images.
		y_test: Test labels.
	"""

	def _summary_lines(name: str, images: np.ndarray, labels: np.ndarray) -> list[str]:
		unique_classes, counts = np.unique(labels, return_counts=True)
		count_map = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
		pixel_min = int(np.min(images)) if images.size else 0
		pixel_max = int(np.max(images)) if images.size else 0
		lines = [
			f"{name}:",
			f"  images: shape={images.shape}, dtype={images.dtype}, min={pixel_min}, max={pixel_max}",
			f"  labels: shape={labels.shape}, dtype={labels.dtype}, unique_classes={len(unique_classes)}",
			"  class counts:",
			"    class | count",
		]
		for class_id in range(10):
			lines.append(f"    {class_id:>5} | {count_map.get(class_id, 0):>5}")
		return lines

	lines = [
		"Dataset summary",
		*_summary_lines("Train set", x_train, y_train),
		*_summary_lines("Test set", x_test, y_test),
	]

	for line in lines:
		LOGGER.info(line)


if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = load_mnist()
	describe_dataset(x_train, y_train, x_test, y_test)

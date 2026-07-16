"""Visualization utilities for image classification datasets."""

from __future__ import annotations

if __package__ is None or __package__ == "":
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_mnist
from predict import load_trained_model, predict_batch


def plot_sample_grid(
	images: np.ndarray,
	labels: np.ndarray,
	predictions: Optional[Sequence[tuple[int, float]] | np.ndarray] = None,
	num_samples: int = 25,
	save_path: Optional[str] = None,
	random_seed: int = 42,
) -> None:
	"""Plot a random grid of grayscale image samples and their labels.

	Args:
		images: Image array of shape ``(n_samples, height, width)``.
		labels: Label array of shape ``(n_samples,)``.
		predictions: Optional sequence of ``(predicted_label, confidence)``
			values aligned with ``images`` and ``labels``.
		num_samples: Number of samples to show. Must be a perfect square.
		save_path: Optional path where the figure is saved at 150 DPI.
		random_seed: Seed used to make random sampling reproducible.

	Raises:
		ValueError: If ``num_samples`` is not a perfect square or exceeds
			the number of available images.
	"""

	total_images = len(images)
	grid_size = int(np.sqrt(num_samples))

	if grid_size * grid_size != num_samples:
		raise ValueError("num_samples must be a perfect square.")
	if num_samples > total_images:
		raise ValueError("num_samples cannot exceed the number of available images.")
	if predictions is not None and len(predictions) != total_images:
		raise ValueError(
			"predictions must have the same length as images and labels: "
			f"got {len(predictions)} predictions for {total_images} images."
		)

	rng = np.random.RandomState(random_seed)
	indices = rng.choice(total_images, size=num_samples, replace=False)
	predicted_labels = None
	prediction_confidences = None
	if predictions is not None:
		predictions_array = np.asarray(predictions, dtype=object)
		predicted_labels = np.asarray([int(item[0]) for item in predictions_array])
		prediction_confidences = np.asarray([float(item[1]) for item in predictions_array])

	fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
	for axis, index in zip(axes.flat, indices):
		axis.imshow(images[index], cmap="gray")
		if predictions is not None:
			predicted_label = int(predicted_labels[index])
			confidence = float(prediction_confidences[index])
			match = predicted_label == int(labels[index])
			axis.set_title(
				f"True: {int(labels[index])}\nPred: {predicted_label} ({confidence:.4f})"
			)
			axis.title.set_color("red" if not match else "black")
		else:
			axis.set_title(str(labels[index]))
		axis.set_xticks([])
		axis.set_yticks([])

	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.show()


def plot_class_distribution(
	labels: np.ndarray,
	title: str,
	save_path: Optional[str] = None,
) -> None:
	"""Plot the class distribution as a bar chart.

	Args:
		labels: Label array.
		title: Figure title.
		save_path: Optional path where the figure is saved at 150 DPI.
	"""

	classes, counts = np.unique(labels, return_counts=True)

	fig, axis = plt.subplots(figsize=(8, 5))
	axis.bar(classes, counts)
	axis.set_xlabel("Class label")
	axis.set_ylabel("Count")
	axis.set_title(title)

	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.show()


def plot_confusion_matrix(
	cm: np.ndarray,
	class_names: list[str],
	save_path: Optional[str] = None,
) -> None:
	"""Plot a confusion matrix heatmap with annotated cell counts.

	Args:
		cm: Confusion matrix array where rows are true classes and columns are
			predicted classes.
		class_names: Ordered class labels used for both axes.
		save_path: Optional path where the figure is saved at 150 DPI.
	"""

	fig, axis = plt.subplots(figsize=(8, 6))
	heatmap = axis.imshow(cm, interpolation="nearest", cmap="Blues")
	fig.colorbar(heatmap, ax=axis)

	axis.set_title("Confusion Matrix")
	axis.set_xlabel("Predicted")
	axis.set_ylabel("True")
	axis.set_xticks(np.arange(len(class_names)))
	axis.set_yticks(np.arange(len(class_names)))
	axis.set_xticklabels(class_names)
	axis.set_yticklabels(class_names)

	threshold = cm.max() / 2.0 if cm.size else 0.0
	for row_index in range(cm.shape[0]):
		for column_index in range(cm.shape[1]):
			value = cm[row_index, column_index]
			axis.text(
				column_index,
				row_index,
				f"{int(value)}",
				ha="center",
				va="center",
				color="white" if value > threshold else "black",
			)

	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.show()


if __name__ == "__main__":
	from pathlib import Path

	(_, _), (x_test, y_test) = load_mnist()
	model_path = Path(__file__).resolve().parent.parent / "models" / "mnist_cnn_final.keras"
	model = load_trained_model(str(model_path))
	predictions = predict_batch(model, x_test)
	plot_sample_grid(x_test, y_test, predictions=predictions, num_samples=25, random_seed=42)
	plot_class_distribution(y_test, title="MNIST Test Set Class Distribution")

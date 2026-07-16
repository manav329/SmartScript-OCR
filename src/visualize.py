"""Visualization utilities for image classification datasets."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_grid(
	images: np.ndarray,
	labels: np.ndarray,
	predictions: Optional[np.ndarray] = None,
	num_samples: int = 25,
	save_path: Optional[str] = None,
	random_seed: int = 42,
) -> None:
	"""Plot a random grid of grayscale image samples and their labels.

	Args:
		images: Image array of shape ``(n_samples, height, width)``.
		labels: Label array of shape ``(n_samples,)``.
		predictions: Optional array of predicted labels.
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

	rng = np.random.RandomState(random_seed)
	indices = rng.choice(total_images, size=num_samples, replace=False)

	fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
	for axis, index in zip(axes.flat, indices):
		axis.imshow(images[index], cmap="gray")
		if predictions is not None:
			axis.set_title(f"True: {labels[index]}, Pred: {predictions[index]}")
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
	from data_loader import load_mnist

	(x_train, y_train), _ = load_mnist()
	plot_sample_grid(x_train, y_train, num_samples=25, random_seed=42)
	plot_class_distribution(y_train, title="MNIST Training Set Class Distribution")

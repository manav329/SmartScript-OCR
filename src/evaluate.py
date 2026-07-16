"""Evaluation utilities for handwritten recognition."""

from __future__ import annotations

import json
import logging
from pathlib import Path

if __package__ is None or __package__ == "":
	import sys

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from config import MODEL_DIR, REPORTS_DIR
from metrics import compute_classification_metrics
from preprocess import encode_labels, normalize_images, reshape_for_cnn
from visualize import plot_confusion_matrix


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _print_classification_table(report: dict, class_names: list[str]) -> None:
	"""Print a compact per-class precision/recall/F1 table."""

	header = f"{'Class':<8}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}"
	print(header)
	print("-" * len(header))
	for class_name in class_names:
		class_report = report[class_name]
		print(
			f"{class_name:<8}"
			f"{class_report['precision']:>12.4f}"
			f"{class_report['recall']:>12.4f}"
			f"{class_report['f1-score']:>12.4f}"
			f"{int(class_report['support']):>12}"
		)


def evaluate_model(
	model_path: str,
	x_test: np.ndarray,
	y_test_onehot: np.ndarray,
	class_names: list[str],
) -> dict:
	"""Evaluate a saved model on a test set and persist the metrics outputs.

	Args:
		model_path: Path to a saved Keras model.
		x_test: Test images.
		y_test_onehot: One-hot encoded test labels.
		class_names: Ordered class labels.

	Returns:
		A dictionary with the classification report, confusion matrix, and
		overall accuracy.
	"""

	model = tf.keras.models.load_model(model_path)
	y_pred_probabilities = model.predict(x_test, verbose=0)
	y_pred = np.argmax(y_pred_probabilities, axis=1)
	y_true = np.argmax(y_test_onehot, axis=1)

	metrics = compute_classification_metrics(y_true, y_pred, class_names)
	report = metrics["report"]

	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	confusion_matrix_path = REPORTS_DIR / "confusion_matrix.png"
	plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=str(confusion_matrix_path))

	metrics_to_save = {
		"report": report,
		"confusion_matrix": metrics["confusion_matrix"].tolist(),
		"overall_accuracy": metrics["overall_accuracy"],
	}
	metrics_json_path = REPORTS_DIR / "test_metrics.json"
	with metrics_json_path.open("w", encoding="utf-8") as file_handle:
		json.dump(metrics_to_save, file_handle, indent=2)

	print(f"Overall test accuracy: {metrics['overall_accuracy']:.4f}")
	_print_classification_table(report, class_names)
	print(f"Saved confusion matrix to {confusion_matrix_path}")
	print(f"Saved metrics JSON to {metrics_json_path}")

	return metrics


if __name__ == "__main__":
	from data_loader import load_mnist

	(_, _), (x_test, y_test) = load_mnist()
	x_test = reshape_for_cnn(normalize_images(x_test))
	y_test_onehot = encode_labels(y_test)
	model_path = MODEL_DIR / "mnist_cnn_final.keras"
	evaluate_model(
		model_path=str(model_path),
		x_test=x_test,
		y_test_onehot=y_test_onehot,
		class_names=[str(index) for index in range(10)],
	)

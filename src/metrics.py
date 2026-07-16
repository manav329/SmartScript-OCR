"""Reusable classification metrics utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_classification_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: list[str],
) -> dict:
	"""Compute classification metrics for integer class labels.

	Args:
		y_true: Ground-truth integer class labels.
		y_pred: Predicted integer class labels.
		class_names: Ordered class names used for the report and confusion matrix.

	Returns:
		A dictionary containing the classification report, confusion matrix, and
		overall accuracy.

	Raises:
		ValueError: If ``y_true`` and ``y_pred`` have different lengths.
	"""

	if len(y_true) != len(y_pred):
		raise ValueError(
			"y_true and y_pred must have the same length: "
			f"got {len(y_true)} and {len(y_pred)}."
		)

	labels = list(range(len(class_names)))
	report = classification_report(
		y_true,
		y_pred,
		labels=labels,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	overall_accuracy = float(np.mean(y_true == y_pred))

	return {
		"report": report,
		"confusion_matrix": cm,
		"overall_accuracy": overall_accuracy,
	}
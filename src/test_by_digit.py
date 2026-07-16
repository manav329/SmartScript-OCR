"""Interactive single-digit evaluation for custom handwritten images."""

from __future__ import annotations

if __package__ is None or __package__ == "":
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import FINAL_MODEL_PATH, IMAGES_TEST_DIR, REPORTS_DIR
from predict import load_trained_model, predict_digit
from utils import load_and_prepare_custom_image


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


MAX_RANDOM_IMAGES = 250


def prompt_for_digit() -> int:
	"""Prompt the user for a digit class and validate the input.

	Returns:
		A validated digit in the range 0 to 9 inclusive.
	"""
	while True:
		user_input = input("Enter a digit to test (0-9): ").strip()
		try:
			digit = int(user_input)
		except ValueError:
			print("Invalid input. Please enter an integer between 0 and 9.")
			continue

		if 0 <= digit <= 9:
			return digit

		print("Digit out of range. Please enter a value from 0 to 9.")


def get_image_paths_for_digit(digit: int, base_dir: Path = IMAGES_TEST_DIR) -> list[Path]:
	"""Collect image files for a selected digit class.

	Args:
		digit: Target digit class in the range 0 to 9.
		base_dir: Root directory that contains per-digit image subfolders.

	Returns:
		Sorted list of image paths with extensions .png, .jpg, or .jpeg.

	Raises:
		FileNotFoundError: If the target digit folder does not exist.
		ValueError: If the folder exists but has no valid image files.
	"""
	target_dir = base_dir / str(digit)
	if not target_dir.exists() or not target_dir.is_dir():
		raise FileNotFoundError(f"Digit folder not found: {target_dir}")

	valid_suffixes = {".png", ".jpg", ".jpeg"}
	image_paths = sorted(
		[
			path
			for path in target_dir.iterdir()
			if path.is_file() and path.suffix.lower() in valid_suffixes
		],
		key=lambda path: path.name.lower(),
	)

	if not image_paths:
		raise ValueError(
			"No valid images found in folder "
			f"{target_dir}. Expected files with extensions: .png, .jpg, .jpeg"
		)

	return image_paths


def evaluate_digit_folder(
	model: tf.keras.Model,
	digit: int,
	image_paths: list[Path],
) -> dict:
	"""Run inference on a digit folder and compute per-digit accuracy.

	Args:
		model: Loaded trained TensorFlow/Keras model.
		digit: Ground-truth digit for all images in ``image_paths``.
		image_paths: Image files to evaluate.

	Returns:
		Dictionary containing aggregate evaluation statistics and per-image
		prediction outputs.
	"""
	predictions: list[int] = []
	confidences: list[float] = []
	skipped = 0

	for image_path in image_paths:
		try:
			prepared_image = load_and_prepare_custom_image(str(image_path))
			predicted_label, confidence = predict_digit(model, prepared_image)
		except Exception as exc:  # pragma: no cover - depends on image/model runtime issues.
			skipped += 1
			LOGGER.warning("Skipping image %s due to error: %s", image_path.name, exc)
			continue

		predictions.append(predicted_label)
		confidences.append(confidence)

	total_images = len(image_paths)
	correct = int(sum(1 for label in predictions if label == digit))
	accuracy = (correct / total_images) if total_images else 0.0

	return {
		"true_digit": digit,
		"total_images": total_images,
		"correct": correct,
		"accuracy": accuracy,
		"predictions": predictions,
		"confidences": confidences,
		"skipped": skipped,
	}


def plot_prediction_distribution(results: dict, save_path: Optional[str] = None) -> None:
	"""Plot predicted-label distribution for one digit folder.

	Args:
		results: Output dictionary from ``evaluate_digit_folder``.
		save_path: Optional output path to save the chart image.
	"""
	predictions = np.asarray(results["predictions"], dtype=int)
	counts = np.bincount(predictions, minlength=10) if predictions.size else np.zeros(10, dtype=int)
	true_digit = int(results["true_digit"])

	bar_colors = ["#9ca3af"] * 10
	bar_colors[true_digit] = "#16a34a"

	plt.figure(figsize=(10, 5))
	plt.bar(np.arange(10), counts, color=bar_colors)
	plt.xticks(np.arange(10))
	plt.xlabel("Predicted Digit")
	plt.ylabel("Count")
	plt.title(
		f"Predictions for digit {true_digit} - Accuracy: {results['accuracy'] * 100:.2f}%"
	)
	plt.tight_layout()

	if save_path:
		save_target = Path(save_path)
		save_target.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_target, dpi=150)

	plt.show()


if __name__ == "__main__":
	try:
		target_digit = prompt_for_digit()

		model = load_trained_model(str(FINAL_MODEL_PATH))

		all_image_paths = get_image_paths_for_digit(target_digit)
		random_generator = np.random.default_rng()
		sample_size = min(MAX_RANDOM_IMAGES, len(all_image_paths))
		selected_indices = random_generator.choice(len(all_image_paths), size=sample_size, replace=False)
		image_paths = [all_image_paths[int(index)] for index in selected_indices]
		target_folder = IMAGES_TEST_DIR / str(target_digit)
		print(
			f"Testing {len(image_paths)} random images from {target_folder} "
			f"(out of {len(all_image_paths)} found) ..."
		)

		results = evaluate_digit_folder(model=model, digit=target_digit, image_paths=image_paths)

		print(
			f"Correct: {results['correct']}/{results['total_images']} "
			f"({results['skipped']} skipped due to load errors)"
		)
		print(
			f"Accuracy for digit {target_digit}: "
			f"{results['accuracy'] * 100:.2f}%"
		)

		report_path = REPORTS_DIR / f"digit_{target_digit}_test_results.png"
		plot_prediction_distribution(results, save_path=str(report_path))
		print(f"Saved chart to {report_path}")

	except Exception:
		LOGGER.exception("Digit folder evaluation failed with an exception.")
		raise SystemExit(1)

"""Training orchestration for the MNIST CNN pipeline."""

from __future__ import annotations

import datetime
import importlib.util
import logging
import sys
from pathlib import Path

import tensorflow as tf

from config import (
	BATCH_SIZE,
	CHECKPOINT_DIR,
	EPOCHS,
	LOGS_DIR,
	MODEL_DIR,
	ensure_directories,
)
from data_loader import load_mnist
from model import build_cnn_model, compile_model
from preprocess import preprocess_pipeline


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
	_handler = logging.StreamHandler()
	_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
	LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def get_callbacks(checkpoint_path: str, log_dir: str) -> list:
	"""Create the callback list used for model training."""

	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=5,
			restore_best_weights=True,
			verbose=1,
		),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			monitor="val_accuracy",
			save_best_only=True,
			verbose=1,
		),
		tf.keras.callbacks.CSVLogger(
			filename=str(Path(log_dir) / "training_log.csv"),
		),
	]

	# TensorFlow may be installed without tensorboard extras in some environments.
	if importlib.util.find_spec("tensorboard") is not None:
		callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
	else:
		LOGGER.warning(
			"TensorBoard package not installed; training will continue without TensorBoard logging."
		)

	return callbacks


def train_model(
	model: tf.keras.Model,
	data: dict,
	epochs: int = EPOCHS,
	batch_size: int = BATCH_SIZE,
	callbacks: list = None,
) -> tf.keras.callbacks.History:
	"""Train the model and return the Keras History object."""

	history = model.fit(
		data["x_train"],
		data["y_train"],
		validation_data=(data["x_val"], data["y_val"]),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=callbacks,
		verbose=1,
	)
	return history


if __name__ == "__main__":
	try:
		ensure_directories()

		(x_train, y_train), (x_test, y_test) = load_mnist()
		processed_data = preprocess_pipeline(x_train, y_train, x_test, y_test)

		run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		checkpoint_path = str(Path(CHECKPOINT_DIR) / f"{run_name}.keras")
		log_dir = str(Path(LOGS_DIR) / run_name)

		model = build_cnn_model()
		model = compile_model(model)

		callbacks = get_callbacks(checkpoint_path=checkpoint_path, log_dir=log_dir)
		history = train_model(model=model, data=processed_data, callbacks=callbacks)

		final_model_path = Path(MODEL_DIR) / "mnist_cnn_final.keras"
		model.save(final_model_path)

		final_training_accuracy = history.history["accuracy"][-1]
		final_validation_accuracy = history.history["val_accuracy"][-1]
		final_training_loss = history.history["loss"][-1]
		final_validation_loss = history.history["val_loss"][-1]

		print("\n=== Final Training Summary ===")
		print(f"Final Training Accuracy: {final_training_accuracy:.4f}")
		print(f"Final Validation Accuracy: {final_validation_accuracy:.4f}")
		print(f"Final Training Loss: {final_training_loss:.4f}")
		print(f"Final Validation Loss: {final_validation_loss:.4f}")
		print(f"Model saved to: {final_model_path}")

	except Exception:
		LOGGER.exception("Training pipeline failed with an exception.")
		sys.exit(1)

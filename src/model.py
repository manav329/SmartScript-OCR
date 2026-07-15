"""Model definition for handwritten recognition."""

from __future__ import annotations

from typing import List

import tensorflow as tf
from keras import layers, models

from config import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE, NUM_CLASSES


def build_cnn_model(
	input_shape: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
	num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
	"""Build a compact CNN for MNIST digit classification.

	The model uses two convolutional feature-extraction blocks followed by a
	small dense classifier, which is sufficient for 28x28 grayscale digits
	without relying on pretrained weights.
	"""

	inputs = layers.Input(shape=input_shape, name="input_image")

	# Learn local edge and stroke patterns from the raw grayscale image.
	x = layers.Conv2D(
		filters=32,
		kernel_size=(3, 3),
		activation="relu",
		padding="same",
		name="conv2d_1",
	)(inputs)
	# Downsample early to reduce spatial size while keeping the strongest features.
	x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1")(x)
	# Learn higher-level combinations of strokes and curves.
	x = layers.Conv2D(
		filters=64,
		kernel_size=(3, 3),
		activation="relu",
		padding="same",
		name="conv2d_2",
	)(x)
	# Reduce feature-map resolution again to keep the classifier lightweight.
	x = layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2")(x)
	# Convert the 2D feature maps into a 1D vector for dense classification.
	x = layers.Flatten(name="flatten")(x)
	# Combine extracted features into a compact latent representation.
	x = layers.Dense(128, activation="relu", name="dense_1")(x)
	# Regularize the classifier to reduce overfitting on MNIST.
	x = layers.Dropout(rate=0.5, name="dropout_1")(x)
	# Produce normalized class probabilities for the 10 digit labels.
	outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

	model = models.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
	return model


def compile_model(model: tf.keras.Model, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
	"""Compile the model with the optimizer, loss, and metric used for MNIST."""

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def get_model_summary_string(model: tf.keras.Model) -> str:
	"""Return the model summary as a string for logging or reporting."""

	summary_lines: List[str] = []
	model.summary(print_fn=summary_lines.append)
	return "\n".join(summary_lines)


if __name__ == "__main__":
	cnn_model = build_cnn_model()
	compile_model(cnn_model)
	print(get_model_summary_string(cnn_model))
	print(f"Total params: {cnn_model.count_params():,}")

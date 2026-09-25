# handwritten-recognition

Project scaffold for handwritten recognition.

# handwritten-recognition

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using TensorFlow and Keras.

## Requirements

- Python 3.8+
- TensorFlow 2.5+
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- TensorBoard (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd handwritten-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the model

```bash
python train_cnn.py
```

This will:
1. Load the MNIST dataset
2. Preprocess the data
3. Train the CNN model
4. Save the trained model to `models/cnn_model.h5`
5. Evaluate the model on the test set
6. Generate evaluation plots

### Run predictions

```bash
python predict.py --image_path path/to/your/image.png
```

This will:
1. Load the trained model
2. Preprocess your image
3. Predict the handwritten digit
4. Display the image with the predicted digit

## Model Architecture

The CNN model consists of the following layers:

- **Input Layer**: Accepts 28x28 grayscale images
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer**: 2x2 pool size
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer**: 2x2 pool size
- **Dropout Layer**: 0.25 dropout rate
- **Flatten Layer**: Flattens the feature maps
- **Dense Layer**: 128 neurons, ReLU activation
- **Output Layer**: 10 neurons, softmax activation

## Results

- Training Accuracy: ~99%
- Test Accuracy: ~99%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- [Manav]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


# Convolutional Neural Network for Traffic Sign Recognition

This project implements a Convolutional Neural Network (CNN) to classify German traffic signs using the [German Traffic Sign Recognition Benchmark (GTSRB)] dataset. The dataset contains over 50,000 labeled images of 43 different types of traffic signs, which are used to train and evaluate the model.

## Project Overview

The goal of this project is to build an image classifier that can recognize German traffic signs. The dataset consists of RGB images, and each image has been resized to 30x30 pixels. The CNN model is designed using TensorFlow and Keras, and the architecture includes multiple convolutional layers, max-pooling, and fully connected layers.

## Dataset

- **Dataset Name**: [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Total Images**: ~50,000
- **Number of Categories**: 43
- **Image Size**: 30x30 pixels
- **Color Mode**: RGB

**NOTE:** I am only including the smaller version of this dataset containing 3 categories. You can download the full data set from [here](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip).
While training with the smaller data set, change `CATEGORIES` to 3.

## Model Architecture

The Convolutional Neural Network (CNN) is structured as follows:

1. **Input Layer**: RGB images of size 30x30x3.
2. **Convolutional Layers**: 
   - 32 filters of size 3x3, followed by ReLU activation.
3. **Max Pooling Layer**: 
   - 2x2 pooling to reduce the dimensionality of the feature map.
4. **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector.
5. **Fully Connected Layers**:
   - Two hidden layers with 128 nodes each, followed by ReLU activation.
6. **Output Layer**: 
   - A dense layer with 43 nodes (one for each category), with softmax activation for multi-class classification.

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metric**: Accuracy
- **Test Size**: 20% of the dataset
- **Number of Epochs**: 10
- **Batch Size**: 32

## Installation

To run the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy

You can install the dependencies by running:

```bash
pip install tensorflow keras numpy 
```

## Output

The model achieves a reasonable accuracy on both the training and test sets after 10 epochs of training. You can modify the number of epochs and other hyperparameters to further improve the model's performance.


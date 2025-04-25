# Handwritten-Digit-Recognition-with-CNN-and-Data-Augmentation

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using TensorFlow/Keras. It incorporates data augmentation to enhance model performance and prevent overfitting. The model is trained, evaluated, and its performance is visualized using various metrics.

## Dataset

The **MNIST dataset** consists of 60,000 28x28 grayscale images for training and 10,000 images for testing, each representing a digit from 0 to 9. It is commonly used for training image classification models.

- **Images**: 28x28 grayscale images
- **Classes**: 10 (Digits 0 to 9)

Dataset source: [MNIST Database]

## Features

- **Data Preprocessing**: Normalized pixel values, reshaped input images for CNN compatibility.
- **Model**: A CNN with two convolutional layers followed by max-pooling layers, and fully connected dense layers.
- **Data Augmentation**: Applied random transformations such as rotation, width/height shifts, and zoom to improve model generalization.
- **Evaluation Metrics**: Model performance is evaluated using accuracy, classification report, and confusion matrix.
- **Visualization**: Training and validation accuracy/loss curves, confusion matrix heatmap.

## Tech Stack

- **TensorFlow**: Deep learning framework used to build and train the CNN model.
- **Keras**: High-level API used for model definition and training.
- **Scikit-learn**: Used for metrics such as confusion matrix and classification report.
- **Matplotlib & Seaborn**: Used for visualizing model performance.

## Installation

Clone the repository and install the required dependencies.

### Clone the repository
git clone https://github.com/Fakiha1407/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
Install Dependencies
For Anaconda, 
We can set up a virtual environment and install dependencies with:
conda create -n tf_env python=3.8
conda activate tf_env
conda install tensorflow matplotlib seaborn scikit-learn
# Usage
After installing the required packages, you can run the training script to train the model on the MNIST dataset:
python train_model.py
## Model Evaluation
After training, the model's performance is evaluated on the test set, and key metrics like accuracy, confusion matrix, and a classification report are printed.

## Visualizations
The following plots will be generated:
Accuracy and Loss curves for both training and validation sets.
Confusion Matrix heatmap showing model predictions vs. true labels.

## Results
Test Accuracy: ~99%
Confusion Matrix: Shows the performance of the model in correctly identifying digits.
Classification Report: Provides precision, recall, and F1-score for each class.

## Contributing
Feel free to fork the repository, contribute via pull requests, or open issues if you encounter bugs or have suggestions for improvement.

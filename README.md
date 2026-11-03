# Neural Network From Scratch

A multi-layer neural network built entirely from scratch using **Python and NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The project demonstrates how neural networks learn through forward propagation, backpropagation, gradient descent, and optimization.

## Overview

This neural network solves a **3-class classification problem** using a generated spiral dataset.

It implements the fundamental components of a neural network manually, including layers, activation functions, loss calculation, gradient computation, and weight optimization.

## Architecture

The network uses the following architecture:

```text
Input Layer (2 neurons)
        |
        v
Dense Layer (64 neurons)
        |
        v
ReLU Activation
        |
        v
Dense Layer (3 neurons)
        |
        v
Softmax Activation
        |
        v
Classification Output
```

## Features

- **Dense Layers:** Matrix multiplication with trainable weights and biases.
- **ReLU Activation:** Introduces non-linearity into the network.
- **Softmax Activation:** Converts output scores into class probabilities.
- **Categorical Cross-Entropy:** Calculates prediction loss.
- **Backpropagation:** Computes gradients for weights, biases, and inputs.
- **Stochastic Gradient Descent (SGD):** Updates model parameters.
- **Momentum:** Improves optimization by incorporating previous updates.
- **Learning Rate Decay:** Gradually reduces the learning rate during training.
- **Validation:** Evaluates the trained network on independently generated data.

## Technologies Used

- Python
- NumPy
- Linear Algebra
- Calculus
- Neural Networks
- Machine Learning Fundamentals

## Installation

Clone the repository:

```bash
git clone https://github.com/blackguy-dev-git/neural-network-from-scratch.git
```

Navigate into the project directory:

```bash
cd neural-network-from-scratch
```

Install the required dependency:

```bash
pip install numpy
```

Run the project:

```bash
python main.py
```

## Dataset

The project uses a custom-generated spiral dataset containing three different classes.

Each sample has two input features, and the network learns to classify the samples into one of three categories.

## Training

The model is trained for 10,001 epochs using:

- Learning rate: 1.0
- Learning rate decay: 0.001
- Momentum: 0.9
- Hidden layer neurons: 64
- Output classes: 3

Training accuracy, loss, and validation performance are printed during execution.

## Learning Objectives

This project was developed to understand the mathematical and computational foundations of neural networks rather than relying entirely on pre-built machine learning frameworks.

Key concepts explored:

- Weights and biases
- Matrix multiplication
- Forward propagation
- Activation functions
- Loss functions
- Backpropagation
- Gradient descent
- Optimization algorithms

## Future Improvements

- Visualize the spiral dataset and decision boundaries.
- Implement additional activation functions.
- Experiment with different network architectures.
- Implement Adam optimization.
- Compare results with PyTorch.

---

**Author:** Moiz Butt

**GitHub:** [blackguy-dev-git](https://github.com/blackguy-dev-git)

*Built as a hands-on exploration of neural networks and the mathematics behind deep learning.*

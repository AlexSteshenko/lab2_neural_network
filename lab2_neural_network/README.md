
# Lab 2: A Neural Network from Scratch

## Overview

This project implements a fully connected neural network from scratch using NumPy to classify handwritten digits from the `load_digits` dataset.

The goal of this laboratory is to understand the fundamentals of neural networks, including forward propagation, cross-entropy loss, backpropagation, numerical gradient checking, and training performance analysis.

## Objectives

- Build a neural network with two hidden layers.
- Implement forward propagation using NumPy.
- Implement the softmax activation function and cross-entropy loss.
- Derive and implement gradients using backpropagation.
- Verify analytical gradients using numerical gradient checking.
- Compare different learning rates and hidden layer widths.
- Measure forward and backward pass execution times.

## Network Architecture

The network has the following architecture:

```text
Input (64 features)
        |
Hidden Layer 1 (W neurons, ReLU)
        |
Hidden Layer 2 (W neurons, ReLU)
        |
Output Layer (10 neurons, Softmax)
```

The hidden layer width `W` is tested with values 16 and 64.

### Weight Matrix Shapes

| Parameter | Shape |
|-----------|-------|
| W1 | (64, W) |
| W2 | (W, W) |
| W3 | (W, 10) |
| b1 | (1, W) |
| b2 | (1, W) |
| b3 | (1, 10) |

Weights are initialized from a normal distribution with mean 0 and standard deviation 0.01. Biases are initialized to zero.

## Dataset

The project uses the handwritten digits dataset provided by scikit-learn.

- Input features: 64 pixel values per image.
- Number of classes: 10 (digits 0–9).
- Input normalization: pixel values are divided by 16.
- Train/test split: 70% training and 30% testing.
- Stratified splitting with `random_state=42`.
- Labels are converted to one-hot encoding.

## Implementation

The implementation includes:

1. Data preprocessing and one-hot encoding.
2. Parameter initialization.
3. Forward propagation using ReLU and Softmax.
4. Cross-entropy loss calculation.
5. Backpropagation for all weights and biases.
6. Numerical gradient checking.
7. Mini-batch gradient descent.
8. Training loss and accuracy tracking.
9. Forward and backward pass timing.

## Experimental Setup

The following configurations are evaluated:

- Learning rates: `0.001`, `0.01`, `0.1`.
- Hidden layer widths: `16`, `64`.
- Mini-batch size: `32`.
- Number of epochs: `50`.

A total of six training configurations are compared.

## Requirements

- Python 3.10 or newer
- NumPy
- scikit-learn
- Matplotlib

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd lab2-neural-network
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python lab2_neural_network.py
```

The script trains the models, performs gradient checking, generates plots, and saves experiment results in the `results/` directory.

## Results

The following files are generated:

| File | Description |
|------|-------------|
| `results/loss_curves.png` | Training loss and test accuracy curves |
| `results/training_summary.csv` | Final loss and accuracy for each configuration |
| `results/gradient_check.csv` | Analytical and numerical gradient comparison |
| `results/timing.txt` | Median forward and backward pass timings |

### Gradient Checking

Numerical gradients are calculated using the central finite difference method.

The gradient check perturbs 10 randomly selected weights and compares the numerical gradients with the analytical gradients computed by backpropagation.

The target is a relative error below `1e-6` for at least 9 out of 10 checked weights.

### Performance Analysis

Forward and backward passes are timed separately over 100 runs. The median execution time is reported for each operation.

The timing comparison helps explain the computational cost of matrix multiplication and gradient calculations.

## Conclusions

This laboratory demonstrates how a neural network can be implemented and trained without deep learning frameworks.

The experiments compare the effects of learning rate and hidden layer width on convergence, classification accuracy, and computational performance.

The final conclusions should be based on the actual training results and gradient checking output.

## Technologies

- Python
- NumPy
- scikit-learn
- Matplotlib

## Author

Student laboratory project.

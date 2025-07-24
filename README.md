# Logistic Regression Implementation from Scratch

**Author:** Sai Yadavalli  
**Version:** 2.2

A complete implementation of logistic regression using gradient descent, built from scratch with NumPy and featuring comprehensive evaluation metrics.

## Overview

This project implements binary logistic regression without using scikit-learn or other machine learning libraries, demonstrating a deep understanding of the underlying mathematical principles and optimization techniques. The implementation includes data preprocessing, model training, evaluation, and visualization capabilities.

## Mathematical Foundation

### Logistic Function (Sigmoid)
The core of logistic regression is the sigmoid function that maps any real number to a probability between 0 and 1:

```
h(x) = 1 / (1 + e^(-w^T * x))
```

Where:
- `w` is the weight vector
- `x` is the feature vector
- `w^T * x` is the linear combination of features

### Cost Function
The logistic regression uses the log-likelihood cost function:

```
J(w) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

Where:
- `m` is the number of training examples
- `y` is the actual label (0 or 1)
- `h(x)` is the predicted probability

### Gradient Descent
Weights are updated using the gradient descent algorithm:

```
w_j := w_j - α * (1/m) * Σ[(h(x^(i)) - y^(i)) * x_j^(i)]
```

Where:
- `α` is the learning rate
- The partial derivative drives the weight updates

## Features

- **Pure NumPy Implementation**: No external ML libraries used
- **Gradient Descent Optimization**: Custom implementation with configurable learning rates
- **Data Standardization**: Optional feature scaling for improved convergence
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Training Visualization**: Real-time cost function plotting
- **Flexible Input**: Works with any CSV dataset
- **Bias Term Handling**: Automatic addition of intercept term

## Key Components

### Core Methods

#### `hwX(w, X)` - Hypothesis Function
Implements the sigmoid function to compute predicted probabilities.

#### `Cost(w, X, y)` - Individual Cost Calculation
Computes the logistic loss for a single training example using the cross-entropy formula.

#### `J(w, X, y, m)` - Average Cost Function
Calculates the mean cost across all training examples, serving as the objective function to minimize.

#### `GD(w, X, y, m, n, alpha)` - Gradient Descent
Updates weights using the computed gradients, with support for feature-specific learning rates.

### Data Processing

#### `ArrayMaker(df, yCol)` - Data Preparation
- Separates features from target variable
- Adds bias term (intercept) as the first column
- Converts pandas DataFrame to NumPy arrays

#### `standardize(df, label)` - Feature Scaling
Implements Z-score normalization:
```
x_scaled = (x - μ) / σ
```

### Training Process

#### `Train(iterations, alpha, standard=False)`
- Iterative weight optimization using gradient descent
- Real-time cost monitoring and logging
- Optional data standardization
- Convergence visualization (last 100 iterations)

### Model Evaluation

#### `evaluate(positive_class, test, train, standard=False)`
Comprehensive model assessment including:
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

## Usage

### Basic Usage
```python
# Load and prepare data
df = pd.read_csv('training_data.csv')
model = LogisticRegression(df, 'target_column')

# Configure training parameters
iterations = 5000
learning_rates = np.full(model.n, 0.000009)

# Train the model
model.Train(iterations, learning_rates, standard=True)

# Evaluate performance
results = model.evaluate(1, 'test_data.csv', 'training_data.csv', standard=True)
```

### Interactive Mode
The implementation includes an interactive command-line interface:
```bash
python logistic_regression.py
```

## Performance Monitoring

The implementation provides:
- **Real-time Cost Tracking**: Monitor convergence during training
- **Iteration-wise Progress**: View cost reduction per iteration
- **Visual Convergence Plot**: Graph showing cost function optimization
- **Detailed Metrics**: Complete classification report with all standard metrics

## Technical Considerations

### Learning Rate Configuration
- Supports feature-specific learning rates
- Default rate: 0.000009 (empirically determined)
- Requires careful tuning for optimal convergence

### Data Preprocessing
- Automatic bias term addition
- Optional standardization for numerical stability
- Handles both continuous and binary features

### Numerical Stability
- Uses NumPy's vectorized operations
- Proper array reshaping for matrix operations
- Handles edge cases in logarithmic calculations

## Requirements

```
pandas>=1.3.0
matplotlib>=3.3.0
numpy>=1.20.0
```

## Educational Value

This implementation demonstrates:
- **Mathematical Understanding**: Complete derivation and implementation of logistic regression
- **Optimization Theory**: Gradient descent from first principles
- **Statistical Evaluation**: Comprehensive model assessment techniques
- **Software Engineering**: Clean, modular, and documented code structure
- **Data Science Pipeline**: End-to-end ML workflow implementation

## Future Enhancements

- [ ] Regularization (L1/L2)
- [ ] Multi-class classification support
- [ ] Advanced optimization algorithms (Adam, RMSprop)
- [ ] Cross-validation implementation
- [ ] Feature selection capabilities

---

This implementation serves as both a practical tool and an educational resource, showcasing the mathematical foundations underlying one of machine learning's most fundamental algorithms.

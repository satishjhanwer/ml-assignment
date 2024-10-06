# Naive Bayes Binary Classification on Ionosphere Dataset

## Problem Statement

We are tasked with implementing a binary classification module using the Ionosphere dataset. The goal is to classify the data points as either 'good' or 'bad', based on radar signals, and compare two implementations: one from scratch and the other using Sklearn.

## Dataset Overview

The dataset contains features obtained from radar signals. Each signal is labeled either 'good' (presence of an object) or 'bad' (empty air). We will split the dataset into an 80:20 ratio for training and testing.

## Approach

### Naive Bayes from Scratch

We implemented a Gaussian Naive Bayes classifier from scratch using the following steps:

1. **Calculate Mean and Variance**: For each class (good/bad), calculate the mean and variance for each feature.
2. **Probability Calculation**: Compute the class-conditional probabilities using Gaussian probability density.
3. **Posterior Calculation**: Use Bayes' Theorem to compute posterior probabilities for each class and make a prediction based on the highest probability.
4. **Evaluation**: Measure accuracy, precision, recall, and F1 score.

### Naive Bayes Using Sklearn

We also used the GaussianNB classifier from the Sklearn library for comparison:

1. **Model Training**: Trained the model using the training data split.
2. **Evaluation**: Predicted on test data and calculated performance metrics.

## Evaluation Metrics

The following metrics were used for performance evaluation:

- **Accuracy**: The percentage of correctly classified instances.
- **Precision**: The percentage of correct positive predictions.
- **Recall**: The percentage of actual positives correctly classified.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix to visualize misclassifications.

## Results

### From Scratch:

- **Accuracy**: `...`
- **Precision**: `...`
- **Recall**: `...`
- **F1 Score**: `...`
- **Confusion Matrix**:
  ```
  [ [TP, FN],
    [FP, TN] ]
  ```

### Sklearn Implementation:

- **Accuracy**: `...`
- **Precision**: `...`
- **Recall**: `...`
- **F1 Score**: `...`
- **Confusion Matrix**:
  ```
  [ [TP, FN],
    [FP, TN] ]
  ```

## Time Comparison

- **Time for Naive Bayes from Scratch**: `... seconds`
- **Time for Sklearn Naive Bayes**: `... seconds`

## Conclusion

The Sklearn implementation is faster and more accurate, benefiting from optimized underlying libraries. However, our custom implementation gives insight into how Naive Bayes works at the foundational level.

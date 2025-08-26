# Chapter 4: Classification Algorithms

This chapter covers various classification algorithms and demonstrates how to implement them using Java. The chapter includes K-Nearest Neighbors, Naive Bayes, Decision Trees, Logistic Regression, and ensemble methods.

## Learning Objectives

- Understand different classification algorithms and their applications
- Implement K-Nearest Neighbors (KNN) classification
- Implement Naive Bayes classification with Gaussian assumptions
- Build and visualize decision trees
- Implement logistic regression for binary classification
- Create ensemble methods using voting
- Evaluate classification performance using multiple metrics
- Build complete classification pipelines

## Chapter Content

### 1. Classification Fundamentals
- **What is Classification?**: Understanding supervised learning for categorical prediction
- **Types of Classification**: Binary vs. multi-class classification
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix

### 2. K-Nearest Neighbors (KNN)
- **Algorithm Overview**: Distance-based classification
- **Implementation**: Euclidean distance calculation and voting
- **Hyperparameters**: Choosing the optimal k value
- **Advantages and Limitations**: When to use KNN

### 3. Naive Bayes Classification
- **Bayesian Approach**: Probability-based classification
- **Gaussian Naive Bayes**: Continuous feature handling
- **Feature Independence**: Understanding the "naive" assumption
- **Implementation**: Prior and likelihood calculations

### 4. Decision Trees
- **Tree Structure**: Nodes, branches, and leaves
- **Splitting Criteria**: Information gain and entropy
- **Tree Building**: Recursive partitioning
- **Pruning**: Preventing overfitting
- **Visualization**: Understanding tree structure

### 5. Logistic Regression
- **Linear Classification**: Extending linear regression for classification
- **Sigmoid Function**: Converting linear output to probabilities
- **Gradient Descent**: Training the model
- **Binary Classification**: Two-class problem solving

### 6. Ensemble Methods
- **Voting**: Combining multiple classifiers
- **Majority Voting**: Simple ensemble approach
- **Advantages**: Improved accuracy and robustness
- **Implementation**: Ensemble classifier framework

### 7. Model Evaluation
- **Cross-validation**: Robust evaluation techniques
- **Metrics Calculation**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Detailed performance analysis
- **Comparison**: Evaluating multiple algorithms

## Project Structure

```
chapter-04-classification/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── aiprogramming/
│                   └── ch04/
│                       ├── ClassificationDemo.java              # Main demonstration class
│                       ├── ClassificationDataPoint.java         # Data point representation
│                       ├── Classifier.java                      # Base classifier interface
│                       ├── KNNClassifier.java                   # K-Nearest Neighbors
│                       ├── NaiveBayesClassifier.java            # Naive Bayes
│                       ├── DecisionTreeClassifier.java          # Decision Tree
│                       ├── LogisticRegressionClassifier.java    # Logistic Regression
│                       ├── EnsembleClassifier.java              # Ensemble methods
│                       ├── ClassificationDataSplitter.java      # Data splitting utility
│                       ├── ClassificationDataSplit.java         # Split data holder
│                       ├── ClassificationEvaluator.java         # Evaluation metrics
│                       └── ClassificationMetrics.java           # Metrics holder
├── pom.xml                                                      # Maven configuration
└── README.md                                                    # This file
```

## Key Classes and Their Purposes

### Core Data Structures
- **`ClassificationDataPoint`**: Represents a single data point with features and class label
- **`Classifier`**: Base interface for all classification algorithms

### Classification Algorithms
- **`KNNClassifier`**: Implements k-nearest neighbors classification
- **`NaiveBayesClassifier`**: Implements Gaussian Naive Bayes
- **`DecisionTreeClassifier`**: Implements decision tree with information gain
- **`LogisticRegressionClassifier`**: Implements binary logistic regression
- **`EnsembleClassifier`**: Combines multiple classifiers using voting

### Utilities
- **`ClassificationDataSplitter`**: Splits data into training and test sets
- **`ClassificationDataSplit`**: Holds training and test data
- **`ClassificationEvaluator`**: Calculates evaluation metrics
- **`ClassificationMetrics`**: Stores evaluation results

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher

## Building and Running

### Compile the Project
```bash
mvn clean compile
```

### Run the Demo
```bash
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch04.ClassificationDemo"
```

### Create Executable JAR
```bash
mvn clean package
```

## Demo Output

The demo demonstrates:

1. **K-Nearest Neighbors**: Distance-based classification with k=3
2. **Naive Bayes**: Probability-based classification using Gaussian assumptions
3. **Decision Trees**: Tree-based classification with information gain splitting
4. **Logistic Regression**: Binary classification using gradient descent
5. **Ensemble Methods**: Combining multiple classifiers using voting
6. **Complete Pipeline**: End-to-end classification with evaluation metrics

## Algorithm Performance

The demo shows performance metrics for each algorithm:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Key Features

- **Modular Design**: Each algorithm implements the same interface
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Ensemble Support**: Easy combination of multiple classifiers
- **Data Splitting**: Stratified splitting to maintain class distribution
- **Extensible**: Easy to add new classification algorithms

## Next Steps

In the following chapters, we'll explore:
- **Chapter 5**: Regression algorithms (Linear, Polynomial, Ridge, Lasso)
- **Chapter 6**: Unsupervised learning (Clustering, Dimensionality Reduction)
- **Chapter 7**: Neural networks and deep learning

The classification foundation built in this chapter will serve as the basis for understanding more advanced machine learning techniques.

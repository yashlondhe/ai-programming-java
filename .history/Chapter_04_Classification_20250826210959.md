# Chapter 4: Supervised Learning - Classification

Welcome to Chapter 4 of **AI Programming with Java**! This chapter focuses on classification, one of the most important and widely-used supervised learning techniques. Classification problems involve predicting discrete categories or classes based on input features. We'll explore various classification algorithms, implement them from scratch in Java, and learn how to evaluate their performance.

## Learning Objectives

After completing this chapter, you will be able to:

- Understand the fundamental concepts of classification problems
- Implement key classification algorithms from scratch in Java
- Apply different classification techniques to real-world problems
- Evaluate classification models using appropriate metrics
- Handle imbalanced datasets and multi-class classification scenarios
- Build a complete classification pipeline from data preprocessing to model deployment
- Compare the strengths and weaknesses of different classification algorithms

## What is Classification?

Classification is a supervised learning task where the goal is to predict a discrete class label for a given input. Unlike regression, which predicts continuous values, classification assigns data points to predefined categories.

### Types of Classification Problems

**Binary Classification:**
- Two possible outcomes (e.g., spam vs. not spam, disease vs. no disease)
- Examples: Email filtering, medical diagnosis, fraud detection

**Multi-class Classification:**
- Three or more mutually exclusive classes (e.g., sentiment analysis: positive, negative, neutral)
- Examples: Image recognition, document categorization, product classification

**Multi-label Classification:**
- Multiple labels can be assigned to a single instance
- Examples: Movie genre tagging, medical symptom diagnosis, news article tagging

### Classification vs. Regression

| Aspect | Classification | Regression |
|--------|----------------|------------|
| Output | Discrete categories | Continuous values |
| Goal | Assign class labels | Predict numeric values |
| Evaluation | Accuracy, Precision, Recall | MAE, MSE, R² |
| Examples | Spam detection, Image recognition | Price prediction, Sales forecasting |

## Classification Algorithms Overview

In this chapter, we'll implement and explore several fundamental classification algorithms:

1. **K-Nearest Neighbors (KNN)** - Instance-based learning
2. **Naive Bayes** - Probabilistic classifier
3. **Decision Trees** - Rule-based classifier
4. **Logistic Regression** - Linear classifier
5. **Ensemble Methods** - Combining multiple classifiers

Each algorithm has unique strengths and is suitable for different types of problems.

## Data Structures for Classification

Before diving into algorithms, let's understand the data structures we'll use throughout this chapter.

### ClassificationDataPoint

Our `ClassificationDataPoint` class represents a single training or test instance:

```java
package com.aiprogramming.ch04;

import java.util.Map;

/**
 * Represents a single data point for classification with features and label
 */
public class ClassificationDataPoint {
    private final Map<String, Double> features;
    private final String label;
    
    public ClassificationDataPoint(Map<String, Double> features, String label) {
        this.features = new HashMap<>(features);
        this.label = label;
    }
    
    public Map<String, Double> getFeatures() {
        return new HashMap<>(features);
    }
    
    public String getLabel() {
        return label;
    }
    
    public double getFeature(String featureName) {
        return features.getOrDefault(featureName, 0.0);
    }
}
```

### Classifier Interface

All our classification algorithms implement the `Classifier` interface:

```java
package com.aiprogramming.ch04;

import java.util.List;
import java.util.Map;

/**
 * Base interface for all classification algorithms
 */
public interface Classifier {
    /**
     * Trains the classifier on the provided training data
     */
    void train(List<ClassificationDataPoint> trainingData);
    
    /**
     * Predicts the class label for a given set of features
     */
    String predict(Map<String, Double> features);
    
    /**
     * Gets the name of the classifier algorithm
     */
    default String getName() {
        return this.getClass().getSimpleName();
    }
}
```

## 1. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is one of the simplest yet effective classification algorithms. It's a non-parametric, instance-based learning method that classifies new data points based on the class of their k nearest neighbors.

### How KNN Works

1. **Store all training data** (lazy learning - no explicit training phase)
2. **For a new prediction:**
   - Calculate the distance between the new point and all training points
   - Find the k nearest neighbors
   - Assign the class based on majority vote among neighbors

### KNN Implementation

```java
package com.aiprogramming.ch04;

import java.util.*;
import java.util.stream.Collectors;

/**
 * K-Nearest Neighbors Classifier Implementation
 */
public class KNNClassifier implements Classifier {
    
    private final int k;
    private List<ClassificationDataPoint> trainingData;
    private Set<String> featureNames;
    
    public KNNClassifier(int k) {
        this.k = k;
        this.trainingData = new ArrayList<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        this.trainingData = new ArrayList<>(trainingData);
        this.featureNames = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (trainingData.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained first");
        }
        
        // Find k nearest neighbors
        List<ClassificationDataPoint> neighbors = findKNearestNeighbors(features);
        
        // Count votes for each class
        Map<String, Long> classVotes = neighbors.stream()
                .collect(Collectors.groupingBy(
                    ClassificationDataPoint::getLabel,
                    Collectors.counting()
                ));
        
        // Return the class with the most votes
        return classVotes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Finds the k nearest neighbors using Euclidean distance
     */
    private List<ClassificationDataPoint> findKNearestNeighbors(Map<String, Double> features) {
        List<DistancePoint> distances = new ArrayList<>();
        
        for (ClassificationDataPoint trainingPoint : trainingData) {
            double distance = calculateDistance(features, trainingPoint.getFeatures());
            distances.add(new DistancePoint(distance, trainingPoint));
        }
        
        // Sort by distance and take top k
        distances.sort(Comparator.comparingDouble(DistancePoint::getDistance));
        return distances.subList(0, Math.min(k, distances.size()))
                .stream()
                .map(DistancePoint::getDataPoint)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * Calculates Euclidean distance between two feature vectors
     */
    private double calculateDistance(Map<String, Double> features1, Map<String, Double> features2) {
        double sum = 0.0;
        for (String feature : featureNames) {
            double value1 = features1.getOrDefault(feature, 0.0);
            double value2 = features2.getOrDefault(feature, 0.0);
            sum += Math.pow(value1 - value2, 2);
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Helper class for storing distance and data point pairs
     */
    private static class DistancePoint {
        private final double distance;
        private final ClassificationDataPoint dataPoint;
        
        public DistancePoint(double distance, ClassificationDataPoint dataPoint) {
            this.distance = distance;
            this.dataPoint = dataPoint;
        }
        
        public double getDistance() {
            return distance;
        }
        
        public ClassificationDataPoint getDataPoint() {
            return dataPoint;
        }
    }
}
```

### KNN Characteristics

**Advantages:**
- Simple to understand and implement
- No assumptions about data distribution
- Works well with small datasets
- Can handle multi-class problems naturally

**Disadvantages:**
- Computationally expensive for large datasets
- Sensitive to irrelevant features
- Sensitive to the choice of k
- Requires feature scaling

**When to Use KNN:**
- Small to medium datasets
- Non-linear decision boundaries
- When interpretability is important
- As a baseline classifier

## 2. Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence. Despite this strong assumption, it often performs well in practice.

### Bayes' Theorem

P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

### Naive Bayes Implementation

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Naive Bayes Classifier Implementation
 */
public class NaiveBayesClassifier implements Classifier {
    
    private Map<String, ClassProbabilities> classProbabilities;
    private Set<String> featureNames;
    
    public NaiveBayesClassifier() {
        this.classProbabilities = new HashMap<>();
        this.featureNames = new HashSet<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Collect all feature names
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
        
        // Calculate feature statistics for each class
        Map<String, List<ClassificationDataPoint>> dataByClass = trainingData.stream()
                .collect(java.util.stream.Collectors.groupingBy(ClassificationDataPoint::getLabel));
        
        // Calculate feature statistics for each class
        for (Map.Entry<String, List<ClassificationDataPoint>> entry : dataByClass.entrySet()) {
            String className = entry.getKey();
            List<ClassificationDataPoint> classData = entry.getValue();
            
            ClassProbabilities classProb = new ClassProbabilities(className, classData.size(), trainingData.size());
            
            // Calculate feature probabilities for this class
            for (String featureName : featureNames) {
                List<Double> featureValues = classData.stream()
                        .map(point -> point.getFeature(featureName))
                        .collect(java.util.stream.Collectors.toList());
                
                FeatureStatistics stats = calculateFeatureStatistics(featureValues);
                classProb.addFeatureStatistics(featureName, stats);
            }
            
            classProbabilities.put(className, classProb);
        }
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (classProbabilities.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        String bestClass = null;
        double bestProbability = Double.NEGATIVE_INFINITY;
        
        // Calculate probability for each class
        for (ClassProbabilities classProb : classProbabilities.values()) {
            double probability = calculateClassProbability(features, classProb);
            
            if (probability > bestProbability) {
                bestProbability = probability;
                bestClass = classProb.getClassName();
            }
        }
        
        return bestClass != null ? bestClass : "unknown";
    }
    
    /**
     * Calculates the probability of a class given the features
     */
    private double calculateClassProbability(Map<String, Double> features, ClassProbabilities classProb) {
        // Start with prior probability (log to avoid numerical underflow)
        double logProbability = Math.log(classProb.getPriorProbability());
        
        // Add likelihood for each feature
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            FeatureStatistics stats = classProb.getFeatureStatistics(featureName);
            
            if (stats != null) {
                double likelihood = calculateGaussianLikelihood(featureValue, stats);
                logProbability += Math.log(likelihood);
            }
        }
        
        return logProbability;
    }
    
    /**
     * Calculates Gaussian likelihood for a feature value
     */
    private double calculateGaussianLikelihood(double value, FeatureStatistics stats) {
        double mean = stats.getMean();
        double stdDev = stats.getStandardDeviation();
        
        if (stdDev == 0) {
            return value == mean ? 1.0 : 0.0;
        }
        
        double exponent = -0.5 * Math.pow((value - mean) / stdDev, 2);
        return (1.0 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    }
    
    /**
     * Calculates mean and standard deviation for a list of values
     */
    private FeatureStatistics calculateFeatureStatistics(List<Double> values) {
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
        
        double stdDev = Math.sqrt(variance);
        
        return new FeatureStatistics(mean, stdDev);
    }
    
    /**
     * Class to store probabilities for a specific class
     */
    private static class ClassProbabilities {
        private final String className;
        private final double priorProbability;
        private final Map<String, FeatureStatistics> featureStats;
        
        public ClassProbabilities(String className, int classCount, int totalCount) {
            this.className = className;
            this.priorProbability = (double) classCount / totalCount;
            this.featureStats = new HashMap<>();
        }
        
        public String getClassName() { return className; }
        public double getPriorProbability() { return priorProbability; }
        
        public void addFeatureStatistics(String featureName, FeatureStatistics stats) {
            featureStats.put(featureName, stats);
        }
        
        public FeatureStatistics getFeatureStatistics(String featureName) {
            return featureStats.get(featureName);
        }
    }
    
    /**
     * Class to store feature statistics
     */
    private static class FeatureStatistics {
        private final double mean;
        private final double standardDeviation;
        
        public FeatureStatistics(double mean, double standardDeviation) {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
        
        public double getMean() { return mean; }
        public double getStandardDeviation() { return standardDeviation; }
    }
}
    

```

### Naive Bayes Characteristics

**Advantages:**
- Fast training and prediction
- Works well with small datasets
- Handles multi-class problems naturally
- Not sensitive to irrelevant features
- Good baseline classifier

**Disadvantages:**
- Strong independence assumption
- Can be outperformed by more sophisticated methods
- Requires smoothing for categorical features

**When to Use Naive Bayes:**
- Text classification
- Spam filtering
- Medical diagnosis
- When features are relatively independent

## 3. Decision Trees

Decision trees create a model that predicts target values by learning simple decision rules inferred from data features. The tree structure represents decisions and their possible consequences.

### How Decision Trees Work

1. **Select the best feature** to split the data (using metrics like Gini impurity or information gain)
2. **Create a branch** for each possible value of that feature
3. **Recursively repeat** the process for each subset
4. **Stop when** a stopping criterion is met (max depth, min samples, pure nodes)

### Decision Tree Implementation

Our `DecisionTreeClassifier` implementation uses information gain to determine the best splits:

```java
package com.aiprogramming.ch04;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Decision Tree Classifier Implementation
 */
public class DecisionTreeClassifier implements Classifier {
    
    private final int maxDepth;
    private TreeNode root;
    private Set<String> featureNames;
    
    public DecisionTreeClassifier(int maxDepth) {
        this.maxDepth = maxDepth;
        this.featureNames = new HashSet<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
        
        // Build the decision tree
        this.root = buildTree(trainingData, 0);
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (root == null) {
            throw new IllegalStateException("Classifier must be trained first");
        }
        
        return predictRecursive(root, features);
    }
    
    /**
     * Recursively builds the decision tree
     */
    private TreeNode buildTree(List<ClassificationDataPoint> data, int depth) {
        // Base cases
        if (data.isEmpty()) {
            return null;
        }
        
        if (depth >= maxDepth) {
            return new TreeNode(getMajorityClass(data), null, null, null, null, true);
        }
        
        // Check if all samples have the same class
        Set<String> uniqueClasses = data.stream()
                .map(ClassificationDataPoint::getLabel)
                .collect(Collectors.toSet());
        
        if (uniqueClasses.size() == 1) {
            return new TreeNode(uniqueClasses.iterator().next(), null, null, null, null, true);
        }
        
        // Find the best split
        SplitInfo bestSplit = findBestSplit(data);
        
        if (bestSplit == null) {
            return new TreeNode(getMajorityClass(data), null, null, null, null, true);
        }
        
        // Split the data
        List<ClassificationDataPoint> leftData = new ArrayList<>();
        List<ClassificationDataPoint> rightData = new ArrayList<>();
        
        for (ClassificationDataPoint point : data) {
            double featureValue = point.getFeature(bestSplit.getFeatureName());
            if (featureValue <= bestSplit.getThreshold()) {
                leftData.add(point);
            } else {
                rightData.add(point);
            }
        }
        
        // Recursively build subtrees
        TreeNode leftChild = buildTree(leftData, depth + 1);
        TreeNode rightChild = buildTree(rightData, depth + 1);
        
        return new TreeNode(null, bestSplit.getFeatureName(), bestSplit.getThreshold(), leftChild, rightChild, false);
    }
    
    private SplitInfo findBestSplit(List<ClassificationDataPoint> data) {
        double bestInformationGain = -1;
        SplitInfo bestSplit = null;
        
        for (String featureName : featureNames) {
            List<Double> featureValues = data.stream()
                    .map(point -> point.getFeature(featureName))
                    .distinct()
                    .sorted()
                    .collect(java.util.stream.Collectors.toList());
            
            // Try different thresholds
            for (int i = 0; i < featureValues.size() - 1; i++) {
                double threshold = (featureValues.get(i) + featureValues.get(i + 1)) / 2.0;
                
                double informationGain = calculateInformationGain(data, featureName, threshold);
                
                if (informationGain > bestInformationGain) {
                    bestInformationGain = informationGain;
                    bestSplit = new SplitInfo(featureName, threshold, informationGain);
                }
            }
        }
        
        return bestSplit;
    }
                

    

    
    /**
     * Gets the majority class from a set of data points
     */
    private String getMajorityClass(List<ClassificationDataPoint> data) {
        return data.stream()
                .collect(Collectors.groupingBy(
                    ClassificationDataPoint::getLabel,
                    Collectors.counting()
                ))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Recursively predicts using the decision tree
     */
    private String predictRecursive(TreeNode node, Map<String, Double> features) {
        if (node.isLeaf()) {
            return node.getPredictedClass();
        }
        
        double featureValue = features.getOrDefault(node.getFeatureName(), 0.0);
        if (featureValue <= node.getThreshold()) {
            return predictRecursive(node.getLeftChild(), features);
        } else {
            return predictRecursive(node.getRightChild(), features);
        }
    }
    
    /**
     * Prints the tree structure for visualization
     */
    public void printTree() {
        printTreeRecursive(root, "", true);
    }
    
    private void printTreeRecursive(TreeNode node, String prefix, boolean isLast) {
        if (node == null) return;
        
        System.out.println(prefix + (isLast ? "└── " : "├── ") + node.toString());
        
        if (!node.isLeaf()) {
            printTreeRecursive(node.getLeftChild(), prefix + (isLast ? "    " : "│   "), false);
            printTreeRecursive(node.getRightChild(), prefix + (isLast ? "    " : "│   "), true);
        }
    }
    
    /**
     * Helper classes for decision tree implementation
     */
    private static class TreeNode {
        private final String predictedClass;
        private final String featureName;
        private final Double threshold;
        private final TreeNode leftChild;
        private final TreeNode rightChild;
        private final boolean isLeaf;
        
        public TreeNode(String predictedClass, String featureName, Double threshold, TreeNode leftChild, TreeNode rightChild, boolean isLeaf) {
            this.predictedClass = predictedClass;
            this.featureName = featureName;
            this.threshold = threshold;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
            this.isLeaf = isLeaf;
        }
        
        // Getters
        public String getPredictedClass() { return predictedClass; }
        public String getFeatureName() { return featureName; }
        public Double getThreshold() { return threshold; }
        public TreeNode getLeftChild() { return leftChild; }
        public TreeNode getRightChild() { return rightChild; }
        public boolean isLeaf() { return isLeaf; }
        
        @Override
        public String toString() {
            if (isLeaf) {
                return "Class: " + predictedClass;
            } else {
                return featureName + " <= " + threshold;
            }
        }
    }
    
    private static class SplitInfo {
        private final String featureName;
        private final double threshold;
        private final double informationGain;
        
        public SplitInfo(String featureName, double threshold, double informationGain) {
            this.featureName = featureName;
            this.threshold = threshold;
            this.informationGain = informationGain;
        }
        
        public String getFeatureName() { return featureName; }
        public double getThreshold() { return threshold; }
        public double getInformationGain() { return informationGain; }
    }
}
```

### Decision Tree Characteristics

**Advantages:**
- Easy to understand and interpret
- Requires little data preparation
- Handles both numerical and categorical features
- Can capture non-linear relationships
- Feature selection is automatic

**Disadvantages:**
- Prone to overfitting
- Can be unstable (small data changes can result in different trees)
- Biased toward features with many levels
- Can create overly complex trees

**When to Use Decision Trees:**
- When interpretability is crucial
- Mixed data types (numerical and categorical)
- Non-linear relationships
- As base learners for ensemble methods

## 4. Logistic Regression

Logistic Regression is a linear classification algorithm that uses the sigmoid function to model the probability of belonging to a class. Despite its name, it's used for classification, not regression.

### How Logistic Regression Works

1. **Linear Combination**: Compute z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
2. **Sigmoid Function**: Apply σ(z) = 1 / (1 + e^(-z)) to get probability
3. **Classification**: Predict class based on probability threshold (usually 0.5)

### Logistic Regression Implementation

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Logistic Regression classifier implementation.
 * Uses gradient descent to train a binary classifier.
 */
public class LogisticRegressionClassifier implements Classifier {
    
    private final double learningRate;
    private final int maxIterations;
    private Map<String, Double> weights;
    private double bias;
    private boolean isTrained = false;
    
    public LogisticRegressionClassifier(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.weights = new HashMap<>();
        this.bias = 0.0;
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Initialize weights for all features
        Set<String> allFeatures = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            allFeatures.addAll(point.getFeatures().keySet());
        }
        
        for (String feature : allFeatures) {
            weights.putIfAbsent(feature, 0.0);
        }
        
        // Convert labels to binary (assuming first class is positive)
        Set<String> uniqueLabels = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            uniqueLabels.add(point.getLabel());
        }
        String positiveClass = uniqueLabels.iterator().next();
        
        // Gradient descent
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double totalLoss = 0.0;
            Map<String, Double> weightGradients = new HashMap<>();
            double biasGradient = 0.0;
            
            // Initialize gradients
            for (String feature : weights.keySet()) {
                weightGradients.put(feature, 0.0);
            }
            
            // Calculate gradients for each training example
            for (ClassificationDataPoint point : trainingData) {
                double prediction = predictProbability(point.getFeatures());
                int actual = point.getLabel().equals(positiveClass) ? 1 : 0;
                
                // Calculate loss
                double loss = -actual * Math.log(prediction + 1e-15) - 
                             (1 - actual) * Math.log(1 - prediction + 1e-15);
                totalLoss += loss;
                
                // Calculate gradients
                double error = prediction - actual;
                biasGradient += error;
                
                for (String feature : weights.keySet()) {
                    double featureValue = point.getFeature(feature);
                    weightGradients.put(feature, weightGradients.get(feature) + error * featureValue);
                }
            }
            
            // Update weights and bias
            bias -= learningRate * biasGradient / trainingData.size();
            
            for (String feature : weights.keySet()) {
                double gradient = weightGradients.get(feature) / trainingData.size();
                weights.put(feature, weights.get(feature) - learningRate * gradient);
            }
            
            // Early stopping if loss is very small
            if (totalLoss / trainingData.size() < 0.01) {
                break;
            }
        }
        
        isTrained = true;
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        double probability = predictProbability(features);
        return probability >= 0.5 ? "positive" : "negative";
    }
    
    /**
     * Predicts the probability of the positive class
     */
    private double predictProbability(Map<String, Double> features) {
        double z = bias;
        
        for (String feature : weights.keySet()) {
            double featureValue = features.getOrDefault(feature, 0.0);
            z += weights.get(feature) * featureValue;
        }
        
        return sigmoid(z);
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    /**
     * Gets the trained weights
     */
    public Map<String, Double> getWeights() {
        return new HashMap<>(weights);
    }
    
    /**
     * Gets the trained bias
     */
    public double getBias() {
        return bias;
    }
}
```

### Logistic Regression Characteristics

**Advantages:**
- Simple and interpretable
- Fast training and prediction
- Provides probability estimates
- Works well with linear decision boundaries
- Less prone to overfitting than complex models

**Disadvantages:**
- Assumes linear relationship between features and log-odds
- May underperform on non-linear problems
- Requires feature scaling for optimal performance
- Limited to binary classification (without extensions)

**When to Use Logistic Regression:**
- Binary classification problems
- When interpretability is important
- Linear decision boundaries
- As a baseline classifier
- When probability estimates are needed

## 5. Ensemble Methods

Ensemble methods combine multiple classifiers to improve prediction accuracy and robustness. We'll implement a simple voting ensemble.

### Ensemble Implementation

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Ensemble classifier that combines multiple classifiers using voting.
 * Implements majority voting for classification.
 */
public class EnsembleClassifier implements Classifier {
    
    private final List<Classifier> classifiers;
    private boolean isTrained = false;
    
    public EnsembleClassifier() {
        this.classifiers = new ArrayList<>();
    }
    
    /**
     * Adds a classifier to the ensemble
     */
    public void addClassifier(Classifier classifier) {
        classifiers.add(classifier);
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (classifiers.isEmpty()) {
            throw new IllegalStateException("No classifiers added to ensemble");
        }
        
        // Train each classifier
        for (Classifier classifier : classifiers) {
            classifier.train(trainingData);
        }
        
        isTrained = true;
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (!isTrained) {
            throw new IllegalStateException("Ensemble must be trained before making predictions");
        }
        
        // Collect predictions from all classifiers
        Map<String, Integer> voteCounts = new HashMap<>();
        
        for (Classifier classifier : classifiers) {
            String prediction = classifier.predict(features);
            voteCounts.put(prediction, voteCounts.getOrDefault(prediction, 0) + 1);
        }
        
        // Return the class with the most votes
        return voteCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Gets the number of classifiers in the ensemble
     */
    public int getClassifierCount() {
        return classifiers.size();
    }
    
    /**
     * Gets the list of classifiers
     */
    public List<Classifier> getClassifiers() {
        return new ArrayList<>(classifiers);
    }
}
```

### Ensemble Characteristics

**Advantages:**
- Improved accuracy through diversity
- Reduced overfitting
- Better generalization
- Robust to individual classifier failures
- Can combine different types of classifiers

**Disadvantages:**
- Increased computational complexity
- More complex to interpret
- Requires more training data
- May not always improve performance

**When to Use Ensemble Methods:**
- When individual classifiers have different strengths
- To improve overall accuracy
- When robustness is important
- With sufficient training data

## Model Evaluation for Classification

Evaluating classification models requires different metrics than regression. Let's implement a comprehensive evaluation framework.

### Classification Metrics

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Evaluates classification performance using various metrics.
 */
public class ClassificationEvaluator {
    
    /**
     * Evaluates classification performance
     */
    public ClassificationMetrics evaluate(List<String> actualLabels, List<String> predictedLabels) {
        if (actualLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("Actual and predicted labels must have the same size");
        }
        
        // Get all unique classes
        Set<String> classes = new HashSet<>();
        classes.addAll(actualLabels);
        classes.addAll(predictedLabels);
        
        // Calculate confusion matrix
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        for (String actualClass : classes) {
            confusionMatrix.put(actualClass, new HashMap<>());
            for (String predictedClass : classes) {
                confusionMatrix.get(actualClass).put(predictedClass, 0);
            }
        }
        
        // Fill confusion matrix
        for (int i = 0; i < actualLabels.size(); i++) {
            String actual = actualLabels.get(i);
            String predicted = predictedLabels.get(i);
            confusionMatrix.get(actual).put(predicted, 
                confusionMatrix.get(actual).get(predicted) + 1);
        }
        
        // Calculate metrics
        double accuracy = calculateAccuracy(confusionMatrix);
        double precision = calculatePrecision(confusionMatrix);
        double recall = calculateRecall(confusionMatrix);
        double f1Score = calculateF1Score(precision, recall);
        
        return new ClassificationMetrics(accuracy, precision, recall, f1Score, confusionMatrix);
    }
    
    private Map<String, Map<String, Integer>> calculateConfusionMatrix(List<String> actualLabels, List<String> predictedLabels) {
        Map<String, Map<String, Integer>> matrix = new HashMap<>();
        
        for (int i = 0; i < actualLabels.size(); i++) {
            String actual = actualLabels.get(i);
            String predicted = predictedLabels.get(i);
            
            matrix.computeIfAbsent(actual, k -> new HashMap<>())
                  .merge(predicted, 1, Integer::sum);
        }
        
        return matrix;
    }
    
    private double calculateAccuracy(List<String> actualLabels, List<String> predictedLabels) {
        int correct = 0;
        for (int i = 0; i < actualLabels.size(); i++) {
            if (actualLabels.get(i).equals(predictedLabels.get(i))) {
                correct++;
            }
        }
        return (double) correct / actualLabels.size();
    }
    
    private double calculatePrecision(Map<String, Map<String, Integer>> confusionMatrix, String className) {
        int truePositive = confusionMatrix.getOrDefault(className, new HashMap<>()).getOrDefault(className, 0);
        int falsePositive = 0;
        
        for (String actualClass : confusionMatrix.keySet()) {
            if (!actualClass.equals(className)) {
                falsePositive += confusionMatrix.get(actualClass).getOrDefault(className, 0);
            }
        }
        
        return (truePositive + falsePositive == 0) ? 0.0 : (double) truePositive / (truePositive + falsePositive);
    }
    
    private double calculateRecall(Map<String, Map<String, Integer>> confusionMatrix, String className) {
        int truePositive = confusionMatrix.getOrDefault(className, new HashMap<>()).getOrDefault(className, 0);
        int falseNegative = 0;
        
        Map<String, Integer> actualClassRow = confusionMatrix.getOrDefault(className, new HashMap<>());
        for (String predictedClass : actualClassRow.keySet()) {
            if (!predictedClass.equals(className)) {
                falseNegative += actualClassRow.get(predictedClass);
            }
        }
        
        return (truePositive + falseNegative == 0) ? 0.0 : (double) truePositive / (truePositive + falseNegative);
    }
    
    private double calculateF1Score(double precision, double recall) {
        return (precision + recall == 0) ? 0.0 : 2 * (precision * recall) / (precision + recall);
    }
}

/**
 * Container class for classification evaluation results
 */
public class ClassificationMetrics {
    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;
    private final Map<String, Map<String, Integer>> confusionMatrix;
    private final Map<String, Double> precisionPerClass;
    private final Map<String, Double> recallPerClass;
    private final Map<String, Double> f1PerClass;
    
    public ClassificationMetrics(double accuracy, double precision, double recall, double f1Score,
                               Map<String, Map<String, Integer>> confusionMatrix,
                               Map<String, Double> precisionPerClass,
                               Map<String, Double> recallPerClass,
                               Map<String, Double> f1PerClass) {
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.confusionMatrix = confusionMatrix;
        this.precisionPerClass = precisionPerClass;
        this.recallPerClass = recallPerClass;
        this.f1PerClass = f1PerClass;
    }
    
    // Getters
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public Map<String, Map<String, Integer>> getConfusionMatrix() { return confusionMatrix; }
    public Map<String, Double> getPrecisionPerClass() { return precisionPerClass; }
    public Map<String, Double> getRecallPerClass() { return recallPerClass; }
    public Map<String, Double> getF1PerClass() { return f1PerClass; }
    
    public void printMetrics() {
        System.out.printf("Accuracy: %.4f%n", accuracy);
        System.out.printf("Precision (Macro): %.4f%n", precision);
        System.out.printf("Recall (Macro): %.4f%n", recall);
        System.out.printf("F1-Score (Macro): %.4f%n", f1Score);
        
        System.out.println("\nPer-Class Metrics:");
        for (String className : precisionPerClass.keySet()) {
            System.out.printf("%s - Precision: %.4f, Recall: %.4f, F1: %.4f%n",
                className, precisionPerClass.get(className), 
                recallPerClass.get(className), f1PerClass.get(className));
        }
    }
}
```

### Understanding Classification Metrics

**Accuracy:** The proportion of correct predictions
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Best for balanced datasets

**Precision:** The proportion of positive identifications that were actually correct
- Formula: TP / (TP + FP)
- Important when false positives are costly

**Recall (Sensitivity):** The proportion of actual positives that were identified correctly
- Formula: TP / (TP + FN)
- Important when false negatives are costly

**F1-Score:** The harmonic mean of precision and recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Good balance between precision and recall

**Confusion Matrix:** A table showing correct vs. predicted classifications
- Rows: Actual classes
- Columns: Predicted classes
- Diagonal elements: Correct predictions

## Ensemble Methods

Ensemble methods combine multiple classifiers to create a stronger predictor than any individual classifier alone.

### Ensemble Classifier Implementation

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Ensemble classifier that combines multiple base classifiers
 */
public class EnsembleClassifier implements Classifier {
    
    private final List<Classifier> baseClassifiers;
    private final VotingStrategy votingStrategy;
    
    public enum VotingStrategy {
        MAJORITY_VOTE,
        WEIGHTED_VOTE
    }
    
    public EnsembleClassifier() {
        this(VotingStrategy.MAJORITY_VOTE);
    }
    
    public EnsembleClassifier(VotingStrategy votingStrategy) {
        this.baseClassifiers = new ArrayList<>();
        this.votingStrategy = votingStrategy;
    }
    
    public void addClassifier(Classifier classifier) {
        baseClassifiers.add(classifier);
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        for (Classifier classifier : baseClassifiers) {
            classifier.train(trainingData);
        }
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (baseClassifiers.isEmpty()) {
            throw new IllegalStateException("No base classifiers added to ensemble");
        }
        
        Map<String, Integer> votes = new HashMap<>();
        
        for (Classifier classifier : baseClassifiers) {
            String prediction = classifier.predict(features);
            votes.merge(prediction, 1, Integer::sum);
        }
        
        return votes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
}
```

## Practical Example: Credit Risk Assessment

Let's build a complete classification pipeline for credit risk assessment:

```java
package com.aiprogramming.ch04;

import java.util.*;

/**
 * Complete classification example: Credit Risk Assessment
 */
public class CreditRiskExample {
    
    public static void main(String[] args) {
        System.out.println("=== Credit Risk Assessment Example ===\n");
        
        // Generate sample credit data
        List<ClassificationDataPoint> creditData = generateCreditData(1000);
        
        // Split data into training and testing sets
        ClassificationDataSplitter splitter = new ClassificationDataSplitter(0.8);
        ClassificationDataSplit split = splitter.split(creditData);
        
        System.out.println("Training samples: " + split.getTrainingData().size());
        System.out.println("Test samples: " + split.getTestData().size());
        
        // Train and evaluate different classifiers
        List<Classifier> classifiers = Arrays.asList(
            new KNNClassifier(5),
            new NaiveBayesClassifier(),
            new DecisionTreeClassifier(5),
            new EnsembleClassifier()
        );
        
        // Configure ensemble
        EnsembleClassifier ensemble = (EnsembleClassifier) classifiers.get(3);
        ensemble.addClassifier(new KNNClassifier(5));
        ensemble.addClassifier(new NaiveBayesClassifier());
        ensemble.addClassifier(new DecisionTreeClassifier(5));
        
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        
        for (Classifier classifier : classifiers) {
            System.out.println("\n" + classifier.getName() + " Results:");
            System.out.println("================================");
            
            // Train the classifier
            long trainStart = System.currentTimeMillis();
            classifier.train(split.getTrainingData());
            long trainTime = System.currentTimeMillis() - trainStart;
            
            // Make predictions
            long predictStart = System.currentTimeMillis();
            List<String> predictions = new ArrayList<>();
            List<String> actuals = new ArrayList<>();
            
            for (ClassificationDataPoint testPoint : split.getTestData()) {
                predictions.add(classifier.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getLabel());
            }
            long predictTime = System.currentTimeMillis() - predictStart;
            
            // Evaluate performance
            ClassificationMetrics metrics = evaluator.evaluate(actuals, predictions);
            metrics.printMetrics();
            
            System.out.printf("Training time: %d ms%n", trainTime);
            System.out.printf("Prediction time: %d ms%n", predictTime);
        }
    }
    
    /**
     * Generates synthetic credit data for demonstration
     */
    private static List<ClassificationDataPoint> generateCreditData(int numSamples) {
        List<ClassificationDataPoint> data = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < numSamples; i++) {
            Map<String, Double> features = new HashMap<>();
            
            // Generate features
            double age = 20 + random.nextGaussian() * 15;
            double income = Math.max(0, 30000 + random.nextGaussian() * 25000);
            double creditHistory = random.nextDouble() * 10; // Years of credit history
            double debtToIncome = random.nextDouble() * 0.8; // Debt to income ratio
            double employmentLength = random.nextDouble() * 20; // Years of employment
            
            features.put("age", age);
            features.put("income", income);
            features.put("credit_history", creditHistory);
            features.put("debt_to_income", debtToIncome);
            features.put("employment_length", employmentLength);
            
            // Determine risk level based on logical rules
            String riskLevel;
            double riskScore = 0.0;
            
            // Age factor
            if (age < 25) riskScore += 0.2;
            else if (age > 40) riskScore -= 0.1;
            
            // Income factor
            if (income < 40000) riskScore += 0.3;
            else if (income > 80000) riskScore -= 0.2;
            
            // Credit history factor
            if (creditHistory < 2) riskScore += 0.3;
            else if (creditHistory > 7) riskScore -= 0.2;
            
            // Debt to income factor
            riskScore += debtToIncome * 0.5;
            
            // Employment length factor
            if (employmentLength < 2) riskScore += 0.2;
            else if (employmentLength > 10) riskScore -= 0.1;
            
            // Add some randomness
            riskScore += (random.nextGaussian() * 0.1);
            
            if (riskScore > 0.4) {
                riskLevel = "high_risk";
            } else if (riskScore > 0.1) {
                riskLevel = "medium_risk";
            } else {
                riskLevel = "low_risk";
            }
            
            data.add(new ClassificationDataPoint(features, riskLevel));
        }
        
        return data;
    }
}
```

## Best Practices for Classification

### 1. Data Preprocessing

```java
/**
 * Data preprocessing utilities for classification
 */
public class ClassificationPreprocessor {
    
    /**
     * Normalize features to [0, 1] range
     */
    public static List<ClassificationDataPoint> normalizeFeatures(List<ClassificationDataPoint> data) {
        // Find min and max for each feature
        Map<String, Double> minValues = new HashMap<>();
        Map<String, Double> maxValues = new HashMap<>();
        
        for (ClassificationDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                minValues.put(feature, Math.min(minValues.getOrDefault(feature, Double.MAX_VALUE), value));
                maxValues.put(feature, Math.max(maxValues.getOrDefault(feature, Double.MIN_VALUE), value));
            }
        }
        
        // Normalize each data point
        List<ClassificationDataPoint> normalizedData = new ArrayList<>();
        for (ClassificationDataPoint point : data) {
            Map<String, Double> normalizedFeatures = new HashMap<>();
            
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                double min = minValues.get(feature);
                double max = maxValues.get(feature);
                
                double normalized = (max - min == 0) ? 0.0 : (value - min) / (max - min);
                normalizedFeatures.put(feature, normalized);
            }
            
            normalizedData.add(new ClassificationDataPoint(normalizedFeatures, point.getLabel()));
        }
        
        return normalizedData;
    }
    
    /**
     * Handle missing values by using mean imputation
     */
    public static List<ClassificationDataPoint> handleMissingValues(List<ClassificationDataPoint> data) {
        // Calculate mean for each feature
        Map<String, Double> featureMeans = calculateFeatureMeans(data);
        
        // Replace missing values with means
        List<ClassificationDataPoint> imputedData = new ArrayList<>();
        for (ClassificationDataPoint point : data) {
            Map<String, Double> imputedFeatures = new HashMap<>();
            
            for (String feature : featureMeans.keySet()) {
                double value = point.getFeatures().getOrDefault(feature, featureMeans.get(feature));
                imputedFeatures.put(feature, value);
            }
            
            imputedData.add(new ClassificationDataPoint(imputedFeatures, point.getLabel()));
        }
        
        return imputedData;
    }
    
    private static Map<String, Double> calculateFeatureMeans(List<ClassificationDataPoint> data) {
        Map<String, List<Double>> featureValues = new HashMap<>();
        
        for (ClassificationDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                featureValues.computeIfAbsent(feature, k -> new ArrayList<>())
                           .add(point.getFeatures().get(feature));
            }
        }
        
        Map<String, Double> means = new HashMap<>();
        for (String feature : featureValues.keySet()) {
            double mean = featureValues.get(feature).stream()
                                    .mapToDouble(Double::doubleValue)
                                    .average()
                                    .orElse(0.0);
            means.put(feature, mean);
        }
        
        return means;
    }
}
```

### 2. Cross-Validation

```java
/**
 * K-Fold Cross-Validation for classification
 */
public class CrossValidator {
    
    public static CrossValidationResults kFoldCrossValidation(
            Classifier classifier, List<ClassificationDataPoint> data, int k) {
        
        List<Double> accuracies = new ArrayList<>();
        List<Double> precisions = new ArrayList<>();
        List<Double> recalls = new ArrayList<>();
        List<Double> f1Scores = new ArrayList<>();
        
        int foldSize = data.size() / k;
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        
        for (int fold = 0; fold < k; fold++) {
            // Create train/test split for this fold
            int startIdx = fold * foldSize;
            int endIdx = (fold == k - 1) ? data.size() : (fold + 1) * foldSize;
            
            List<ClassificationDataPoint> testFold = data.subList(startIdx, endIdx);
            List<ClassificationDataPoint> trainFold = new ArrayList<>();
            trainFold.addAll(data.subList(0, startIdx));
            trainFold.addAll(data.subList(endIdx, data.size()));
            
            // Train and evaluate
            classifier.train(trainFold);
            
            List<String> predictions = new ArrayList<>();
            List<String> actuals = new ArrayList<>();
            
            for (ClassificationDataPoint testPoint : testFold) {
                predictions.add(classifier.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getLabel());
            }
            
            ClassificationMetrics metrics = evaluator.evaluate(actuals, predictions);
            
            accuracies.add(metrics.getAccuracy());
            precisions.add(metrics.getPrecision());
            recalls.add(metrics.getRecall());
            f1Scores.add(metrics.getF1Score());
        }
        
        return new CrossValidationResults(accuracies, precisions, recalls, f1Scores);
    }
}

/**
 * Cross-validation results container
 */
public class CrossValidationResults {
    private final List<Double> accuracies;
    private final List<Double> precisions;
    private final List<Double> recalls;
    private final List<Double> f1Scores;
    
    public CrossValidationResults(List<Double> accuracies, List<Double> precisions, 
                                List<Double> recalls, List<Double> f1Scores) {
        this.accuracies = accuracies;
        this.precisions = precisions;
        this.recalls = recalls;
        this.f1Scores = f1Scores;
    }
    
    public double getMeanAccuracy() {
        return accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdAccuracy() {
        double mean = getMeanAccuracy();
        double variance = accuracies.stream()
                .mapToDouble(acc -> Math.pow(acc - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    // Similar methods for precision, recall, and F1-score...
    
    public void printResults() {
        System.out.printf("Cross-Validation Results (Mean ± Std):%n");
        System.out.printf("Accuracy: %.4f ± %.4f%n", getMeanAccuracy(), getStdAccuracy());
        System.out.printf("Precision: %.4f ± %.4f%n", getMeanPrecision(), getStdPrecision());
        System.out.printf("Recall: %.4f ± %.4f%n", getMeanRecall(), getStdRecall());
        System.out.printf("F1-Score: %.4f ± %.4f%n", getMeanF1Score(), getStdF1Score());
    }
}
```

## Running the Examples

To run the classification examples, navigate to the chapter-04-classification directory and execute:

```bash
cd chapter-04-classification
javac -d . src/main/java/com/aiprogramming/ch04/*.java
java com.aiprogramming.ch04.ClassificationDemo
```

This will demonstrate all the classification algorithms we've implemented.

## Summary

In this chapter, we've covered the fundamental concepts of classification and implemented several key algorithms from scratch:

1. **K-Nearest Neighbors (KNN)** - Simple, instance-based learning
2. **Naive Bayes** - Probabilistic classifier with independence assumption
3. **Decision Trees** - Rule-based, interpretable classifier
4. **Ensemble Methods** - Combining multiple classifiers for better performance

We've also learned about:
- Classification evaluation metrics (accuracy, precision, recall, F1-score)
- Cross-validation techniques
- Data preprocessing for classification
- Practical implementation considerations

### Key Takeaways

1. **Choose the right algorithm** based on your data and requirements
2. **Evaluate properly** using appropriate metrics for your use case
3. **Preprocess your data** to improve algorithm performance
4. **Use cross-validation** to get reliable performance estimates
5. **Consider ensemble methods** for improved accuracy
6. **Balance interpretability and performance** based on your needs

### Next Steps

In the next chapter, we'll explore **Regression**, another fundamental supervised learning technique that predicts continuous values instead of discrete classes. We'll implement linear regression, polynomial regression, and other regression techniques, and learn how to evaluate regression models.

The concepts you've learned in this chapter about supervised learning, evaluation, and model selection will be directly applicable to regression problems as well.

## Exercises

1. **Implement Logistic Regression** from scratch using gradient descent
2. **Add feature selection** to improve classifier performance
3. **Implement Support Vector Machine** classifier
4. **Create a text classification** system using Naive Bayes
5. **Build a Random Forest** classifier using multiple decision trees
6. **Implement stratified cross-validation** to handle imbalanced datasets
7. **Add hyperparameter tuning** using grid search or random search

## Further Reading

- *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman
- *Pattern Recognition and Machine Learning* by Christopher Bishop
- *Hands-On Machine Learning* by Aurélien Géron
- Java ML libraries: Weka, Smile, Tribuo
- Scikit-learn documentation for algorithm comparison
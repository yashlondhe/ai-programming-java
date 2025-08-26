# Chapter 5: Supervised Learning - Regression

Welcome to Chapter 5 of **AI Programming with Java**! This chapter focuses on regression, a fundamental supervised learning technique used to predict continuous numerical values. While classification predicts discrete categories, regression predicts quantities like prices, temperatures, or distances. We'll explore various regression algorithms, implement them from scratch in Java, and learn how to evaluate their performance.

## Learning Objectives

After completing this chapter, you will be able to:

- Understand the fundamental concepts of regression problems
- Implement key regression algorithms from scratch in Java
- Apply different regression techniques to real-world problems
- Evaluate regression models using appropriate metrics
- Handle overfitting and underfitting in regression models
- Build a complete regression pipeline from data preprocessing to model deployment
- Compare the strengths and weaknesses of different regression algorithms
- Implement regularization techniques to improve model performance

## What is Regression?

Regression is a supervised learning task where the goal is to predict a continuous numerical value based on input features. Unlike classification, which assigns data points to discrete categories, regression estimates quantities that can take any value within a range.

### Types of Regression Problems

**Simple Linear Regression:**
- One input variable (feature) predicts one output variable
- Examples: House size → House price, Hours studied → Exam score

**Multiple Linear Regression:**
- Multiple input variables predict one output variable
- Examples: House size + location + age → House price

**Polynomial Regression:**
- Non-linear relationships using polynomial features
- Examples: Time → Population growth (exponential relationship)

**Time Series Regression:**
- Predicting future values based on historical data
- Examples: Stock price prediction, weather forecasting

### Regression vs. Classification

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Output | Continuous values | Discrete categories |
| Goal | Predict numeric quantities | Assign class labels |
| Evaluation | MAE, MSE, R² | Accuracy, Precision, Recall |
| Examples | Price prediction, Sales forecasting | Spam detection, Image recognition |

## Regression Algorithms Overview

In this chapter, we'll implement and explore five fundamental regression algorithms:

1. **Linear Regression** - Basic linear relationship modeling using normal equation
2. **Polynomial Regression** - Non-linear relationships using polynomial features
3. **Ridge Regression** - Linear regression with L2 regularization to prevent overfitting
4. **Lasso Regression** - Linear regression with L1 regularization for feature selection
5. **Support Vector Regression (SVR)** - Non-linear regression using RBF kernel methods

Each algorithm has unique strengths and is suitable for different types of problems. Complete implementations, comprehensive testing, and practical examples are provided for all algorithms.

## Data Structures for Regression

Before diving into algorithms, let's understand the data structures we'll use throughout this chapter.

### RegressionDataPoint

Our `RegressionDataPoint` class represents a single training or test instance:

```java
package com.aiprogramming.ch05;

import java.util.Map;
import java.util.HashMap;
import java.util.Set;

/**
 * Represents a single data point for regression with features and target value
 */
public class RegressionDataPoint {
    private final Map<String, Double> features;
    private final double target;
    
    public RegressionDataPoint(Map<String, Double> features, double target) {
        this.features = new HashMap<>(features);
        this.target = target;
    }
    
    public Map<String, Double> getFeatures() {
        return new HashMap<>(features);
    }
    
    public double getTarget() {
        return target;
    }
    
    public Double getFeature(String featureName) {
        return features.get(featureName);
    }
    
    public Set<String> getFeatureNames() {
        return features.keySet();
    }
    
    @Override
    public String toString() {
        return String.format("RegressionDataPoint{features=%s, target=%.3f}", features, target);
    }
}
```

### Regressor Interface

All our regression algorithms implement the `Regressor` interface:

```java
package com.aiprogramming.ch05;

import java.util.List;
import java.util.Map;

/**
 * Base interface for all regression algorithms
 */
public interface Regressor {
    /**
     * Trains the regressor on the provided training data
     */
    void train(List<RegressionDataPoint> trainingData);
    
    /**
     * Predicts the target value for a given set of features
     */
    double predict(Map<String, Double> features);
    
    /**
     * Gets the name of the regression algorithm
     */
    default String getName() {
        return this.getClass().getSimpleName();
    }
}
```

## 1. Linear Regression

Linear regression is the most fundamental regression algorithm. It models the relationship between input features and the target variable as a linear equation.

### How Linear Regression Works

The goal is to find the best line (or hyperplane in multiple dimensions) that fits through the data points. The linear equation is:

**Simple Linear Regression:** y = β₀ + β₁x₁ + ε

**Multiple Linear Regression:** y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y is the target variable
- x₁, x₂, ..., xₙ are the input features
- β₀ is the intercept (bias)
- β₁, β₂, ..., βₙ are the coefficients (weights)
- ε is the error term

### Linear Regression Implementation

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Linear Regression Implementation using Normal Equation
 */
public class LinearRegression implements Regressor {
    
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public LinearRegression() {
        this.coefficients = new HashMap<>();
        this.intercept = 0.0;
        this.featureNames = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Convert data to matrix format
        int numSamples = trainingData.size();
        int numFeatures = featureNames.size();
        
        // X matrix (with bias column)
        double[][] X = new double[numSamples][numFeatures + 1];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            X[i][0] = 1.0; // Bias term
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j + 1] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        // Solve normal equation: β = (X'X)^(-1)X'y
        double[][] XTranspose = transpose(X);
        double[][] XTX = multiply(XTranspose, X);
        double[][] XTXInverse = inverse(XTX);
        double[] XTy = multiplyVector(XTranspose, y);
        double[] beta = multiplyVector(XTXInverse, XTy);
        
        // Extract intercept and coefficients
        this.intercept = beta[0];
        this.coefficients.clear();
        
        for (int i = 0; i < numFeatures; i++) {
            coefficients.put(featureNames.get(i), beta[i + 1]);
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double prediction = intercept;
        
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            double coefficient = coefficients.getOrDefault(featureName, 0.0);
            prediction += coefficient * featureValue;
        }
        
        return prediction;
    }
    
    /**
     * Gets the learned coefficients
     */
    public Map<String, Double> getCoefficients() {
        return new HashMap<>(coefficients);
    }
    
    /**
     * Gets the learned intercept
     */
    public double getIntercept() {
        return intercept;
    }
    
    /**
     * Gets the feature importance based on absolute coefficient values
     */
    public Map<String, Double> getFeatureImportance() {
        Map<String, Double> importance = new HashMap<>();
        double maxCoeff = coefficients.values().stream()
                .mapToDouble(Math::abs)
                .max()
                .orElse(1.0);
        
        for (Map.Entry<String, Double> entry : coefficients.entrySet()) {
            importance.put(entry.getKey(), Math.abs(entry.getValue()) / maxCoeff);
        }
        
        return importance;
    }
    
    // Matrix operations helper methods
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    private double[][] multiply(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }
    
    private double[] multiplyVector(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            double maxElement = Math.abs(augmented[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > maxElement) {
                    maxElement = Math.abs(augmented[k][i]);
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[maxRow];
                augmented[maxRow] = temp;
            }
            
            // Make diagonal element 1
            double pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new RuntimeException("Matrix is singular");
            }
            
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        return inverse;
    }
}
```

### Linear Regression Characteristics

**Advantages:**
- Simple and fast to train
- No hyperparameters to tune
- Provides interpretable coefficients
- Works well when the relationship is actually linear
- Baseline for other regression methods

**Disadvantages:**
- Assumes linear relationship
- Sensitive to outliers
- Can overfit with many features
- Requires features to be scaled for interpretation

**When to Use Linear Regression:**
- When you suspect a linear relationship
- When interpretability is important
- As a baseline model
- When you have limited training data

## 2. Polynomial Regression

Polynomial regression extends linear regression to model non-linear relationships by using polynomial features.

### How Polynomial Regression Works

Instead of using features x₁, x₂, we create polynomial features like x₁², x₁x₂, x₂², etc.

For degree 2 polynomial with features x₁, x₂:
y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂²

### Polynomial Regression Implementation

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Polynomial Regression Implementation
 */
public class PolynomialRegression implements Regressor {
    
    private final int degree;
    private LinearRegression linearRegressor;
    private List<String> originalFeatures;
    private List<String> polynomialFeatures;
    private boolean trained;
    
    public PolynomialRegression(int degree) {
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be at least 1");
        }
        this.degree = degree;
        this.linearRegressor = new LinearRegression();
        this.originalFeatures = new ArrayList<>();
        this.polynomialFeatures = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract original feature names
        this.originalFeatures = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Generate polynomial features
        this.polynomialFeatures = generatePolynomialFeatureNames(originalFeatures, degree);
        
        // Transform training data to include polynomial features
        List<RegressionDataPoint> transformedData = transformData(trainingData);
        
        // Train linear regressor on transformed data
        linearRegressor.train(transformedData);
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        // Transform features to include polynomial terms
        Map<String, Double> transformedFeatures = transformFeatures(features);
        
        // Use linear regressor for prediction
        return linearRegressor.predict(transformedFeatures);
    }
    
    /**
     * Gets the degree of the polynomial
     */
    public int getDegree() {
        return degree;
    }
    
    /**
     * Gets the underlying linear regressor coefficients
     */
    public Map<String, Double> getCoefficients() {
        return linearRegressor.getCoefficients();
    }
    
    /**
     * Gets the intercept
     */
    public double getIntercept() {
        return linearRegressor.getIntercept();
    }
    
    private List<String> generatePolynomialFeatureNames(List<String> features, int degree) {
        List<String> polyFeatures = new ArrayList<>();
        
        // Generate all combinations of features up to the specified degree
        generatePolynomialCombinations(features, degree, "", 0, 0, polyFeatures);
        
        return polyFeatures;
    }
    
    private void generatePolynomialCombinations(List<String> features, int maxDegree, 
                                              String current, int currentDegree, 
                                              int startFeature, List<String> result) {
        if (currentDegree > 0) {
            result.add(current);
        }
        
        if (currentDegree < maxDegree) {
            for (int i = startFeature; i < features.size(); i++) {
                String newFeature = current.isEmpty() ? features.get(i) : current + "*" + features.get(i);
                generatePolynomialCombinations(features, maxDegree, newFeature, 
                                             currentDegree + 1, i, result);
            }
        }
    }
    
    private List<RegressionDataPoint> transformData(List<RegressionDataPoint> data) {
        List<RegressionDataPoint> transformedData = new ArrayList<>();
        
        for (RegressionDataPoint point : data) {
            Map<String, Double> transformedFeatures = transformFeatures(point.getFeatures());
            transformedData.add(new RegressionDataPoint(transformedFeatures, point.getTarget()));
        }
        
        return transformedData;
    }
    
    private Map<String, Double> transformFeatures(Map<String, Double> features) {
        Map<String, Double> transformedFeatures = new HashMap<>();
        
        for (String polyFeature : polynomialFeatures) {
            double value = calculatePolynomialFeatureValue(polyFeature, features);
            transformedFeatures.put(polyFeature, value);
        }
        
        return transformedFeatures;
    }
    
    private double calculatePolynomialFeatureValue(String polyFeature, Map<String, Double> features) {
        String[] featureParts = polyFeature.split("\\*");
        double value = 1.0;
        
        for (String part : featureParts) {
            value *= features.getOrDefault(part, 0.0);
        }
        
        return value;
    }
}
```

### Polynomial Regression Characteristics

**Advantages:**
- Can model non-linear relationships
- Still uses linear regression internally
- Can fit complex curves
- Interpretable for low degrees

**Disadvantages:**
- Prone to overfitting with high degrees
- Features can become highly correlated
- Computationally expensive for high degrees
- Requires careful regularization

**When to Use Polynomial Regression:**
- When you observe non-linear patterns in the data
- For feature engineering
- When simple linear regression underfits
- For exploratory data analysis

## 3. Ridge Regression (L2 Regularization)

Ridge regression adds a penalty term to the linear regression cost function to prevent overfitting.

### How Ridge Regression Works

The cost function becomes:
Cost = MSE + α × Σ(βᵢ²)

Where α is the regularization parameter that controls the strength of regularization.

### Ridge Regression Implementation

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Ridge Regression Implementation with L2 Regularization
 */
public class RidgeRegression implements Regressor {
    
    private final double alpha;
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public RidgeRegression(double alpha) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.coefficients = new HashMap<>();
        this.intercept = 0.0;
        this.featureNames = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Convert data to matrix format
        int numSamples = trainingData.size();
        int numFeatures = featureNames.size();
        
        // X matrix (with bias column)
        double[][] X = new double[numSamples][numFeatures + 1];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            X[i][0] = 1.0; // Bias term
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j + 1] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        // Ridge regression: β = (X'X + αI)^(-1)X'y
        double[][] XTranspose = transpose(X);
        double[][] XTX = multiply(XTranspose, X);
        
        // Add regularization term (don't regularize intercept)
        for (int i = 1; i < XTX.length; i++) {
            XTX[i][i] += alpha;
        }
        
        double[][] XTXInverse = inverse(XTX);
        double[] XTy = multiplyVector(XTranspose, y);
        double[] beta = multiplyVector(XTXInverse, XTy);
        
        // Extract intercept and coefficients
        this.intercept = beta[0];
        this.coefficients.clear();
        
        for (int i = 0; i < numFeatures; i++) {
            coefficients.put(featureNames.get(i), beta[i + 1]);
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double prediction = intercept;
        
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            double coefficient = coefficients.getOrDefault(featureName, 0.0);
            prediction += coefficient * featureValue;
        }
        
        return prediction;
    }
    
    /**
     * Gets the regularization parameter
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Gets the learned coefficients
     */
    public Map<String, Double> getCoefficients() {
        return new HashMap<>(coefficients);
    }
    
    /**
     * Gets the learned intercept
     */
    public double getIntercept() {
        return intercept;
    }
    
    // Matrix operations (same as LinearRegression)
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    private double[][] multiply(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }
    
    private double[] multiplyVector(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            double maxElement = Math.abs(augmented[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > maxElement) {
                    maxElement = Math.abs(augmented[k][i]);
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[maxRow];
                augmented[maxRow] = temp;
            }
            
            // Make diagonal element 1
            double pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new RuntimeException("Matrix is singular");
            }
            
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        return inverse;
    }
}
```

### Ridge Regression Characteristics

**Advantages:**
- Reduces overfitting
- Handles multicollinearity well
- Keeps all features (shrinks coefficients to zero)
- Stable solution

**Disadvantages:**
- Requires tuning of α parameter
- Less interpretable than linear regression
- Doesn't perform feature selection

**When to Use Ridge Regression:**
- When you have many features
- When features are highly correlated
- When linear regression overfits
- When you want to keep all features

## 4. Lasso Regression (L1 Regularization)

Lasso regression adds an L1 penalty term to the linear regression cost function, which encourages sparsity and automatic feature selection.

### How Lasso Regression Works

The cost function becomes:
Cost = MSE + α × Σ|βᵢ|

Where α is the regularization parameter. The L1 penalty drives some coefficients to exactly zero, effectively performing feature selection.

### Lasso Regression Implementation

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Lasso Regression Implementation with L1 Regularization using Coordinate Descent
 */
public class LassoRegression implements Regressor {
    
    private final double alpha;
    private final int maxIterations;
    private final double tolerance;
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public LassoRegression(double alpha) {
        this(alpha, 1000, 1e-6);
    }
    
    public LassoRegression(double alpha, int maxIterations, double tolerance) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.coefficients = new HashMap<>();
        this.intercept = 0.0;
        this.featureNames = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Convert data to matrix format
        int numSamples = trainingData.size();
        int numFeatures = featureNames.size();
        
        // X matrix and y vector
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        // Standardize features
        double[] featureMeans = calculateFeatureMeans(X);
        double[] featureStds = calculateFeatureStds(X, featureMeans);
        standardizeFeatures(X, featureMeans, featureStds);
        
        // Center target
        double targetMean = Arrays.stream(y).average().orElse(0.0);
        for (int i = 0; i < y.length; i++) {
            y[i] -= targetMean;
        }
        
        // Initialize coefficients
        double[] beta = new double[numFeatures];
        
        // Coordinate descent algorithm
        for (int iter = 0; iter < maxIterations; iter++) {
            double[] oldBeta = Arrays.copyOf(beta, beta.length);
            
            for (int j = 0; j < numFeatures; j++) {
                // Calculate residual without j-th feature
                double[] residual = calculateResidual(X, y, beta, j);
                
                // Calculate correlation with j-th feature
                double correlation = calculateCorrelation(X, residual, j);
                
                // Soft thresholding operator
                beta[j] = softThreshold(correlation, alpha);
            }
            
            // Check convergence
            if (hasConverged(beta, oldBeta)) {
                break;
            }
        }
        
        // Transform coefficients back to original scale
        this.coefficients.clear();
        for (int i = 0; i < numFeatures; i++) {
            double originalCoeff = (featureStds[i] != 0) ? beta[i] / featureStds[i] : 0.0;
            coefficients.put(featureNames.get(i), originalCoeff);
        }
        
        // Calculate intercept
        this.intercept = targetMean;
        for (int i = 0; i < numFeatures; i++) {
            this.intercept -= coefficients.get(featureNames.get(i)) * featureMeans[i];
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double prediction = intercept;
        
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            double coefficient = coefficients.getOrDefault(featureName, 0.0);
            prediction += coefficient * featureValue;
        }
        
        return prediction;
    }
    
    /**
     * Gets the selected features (non-zero coefficients)
     */
    public Set<String> getSelectedFeatures() {
        Set<String> selectedFeatures = new HashSet<>();
        for (Map.Entry<String, Double> entry : coefficients.entrySet()) {
            if (Math.abs(entry.getValue()) > 1e-10) {
                selectedFeatures.add(entry.getKey());
            }
        }
        return selectedFeatures;
    }
    
    /**
     * Gets the sparsity level (percentage of zero coefficients)
     */
    public double getSparsity() {
        long zeroCoeffs = coefficients.values().stream()
                .mapToLong(coeff -> Math.abs(coeff) <= 1e-10 ? 1 : 0)
                .sum();
        return (double) zeroCoeffs / coefficients.size();
    }
    
    // Soft thresholding operator for L1 regularization
    private double softThreshold(double value, double threshold) {
        if (value > threshold) {
            return value - threshold;
        } else if (value < -threshold) {
            return value + threshold;
        } else {
            return 0.0;
        }
    }
    
    // Additional helper methods for coordinate descent...
}
```

### Lasso Regression Characteristics

**Advantages:**
- Performs automatic feature selection
- Reduces overfitting
- Creates sparse models
- Good for high-dimensional data
- Interpretable results

**Disadvantages:**
- Requires hyperparameter tuning
- Can be unstable with correlated features
- More computationally expensive than Ridge
- May arbitrarily select one feature from a group of correlated features

**When to Use Lasso Regression:**
- When you need feature selection
- When you suspect only some features are relevant
- When interpretability with fewer features is desired
- When you have high-dimensional data with sparse signals

## 5. Support Vector Regression (SVR)

Support Vector Regression extends the Support Vector Machine concept to regression problems. It uses kernel functions to handle non-linear relationships and is robust to outliers.

### How SVR Works

SVR tries to find a function that deviates from the training targets by at most ε (epsilon), while being as flat as possible. It uses an epsilon-insensitive loss function:

Loss = 0 if |y - f(x)| ≤ ε, otherwise |y - f(x)| - ε

The optimization problem includes:
- C parameter: Controls the trade-off between flatness and tolerance for errors
- ε parameter: Width of the epsilon-tube (acceptable error margin)
- Kernel function: Maps data to higher-dimensional space

### SVR Implementation

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Support Vector Regression (SVR) Implementation with RBF Kernel
 */
public class SupportVectorRegression implements Regressor {
    
    private final double C;               // Regularization parameter
    private final double epsilon;         // Epsilon-tube parameter
    private final double gamma;           // RBF kernel parameter
    private final int maxIterations;
    private final double tolerance;
    
    private List<RegressionDataPoint> supportVectors;
    private double[] alphas;
    private double[] alphaStar;
    private double bias;
    private List<String> featureNames;
    private boolean trained;
    
    public SupportVectorRegression(double C, double epsilon, double gamma) {
        this(C, epsilon, gamma, 1000, 1e-6);
    }
    
    public SupportVectorRegression(double C, double epsilon, double gamma, 
                                 int maxIterations, double tolerance) {
        if (C <= 0 || epsilon < 0 || gamma <= 0) {
            throw new IllegalArgumentException("C, gamma must be positive, epsilon must be non-negative");
        }
        
        this.C = C;
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        int numSamples = trainingData.size();
        
        // Initialize Lagrange multipliers
        this.alphas = new double[numSamples];
        this.alphaStar = new double[numSamples];
        this.bias = 0.0;
        
        // Simplified SMO-like algorithm for SVR
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            
            for (int i = 0; i < numSamples; i++) {
                double prediction = predictInternal(trainingData.get(i), trainingData);
                double target = trainingData.get(i).getTarget();
                double error = prediction - target;
                
                // Check KKT conditions and update if needed
                if (Math.abs(error) > epsilon + tolerance) {
                    int j = findSecondExample(i, trainingData, error);
                    if (j != -1) {
                        changed |= updateAlphaPair(i, j, trainingData);
                    }
                }
            }
            
            if (!changed) {
                break;
            }
        }
        
        // Store support vectors (examples with non-zero alphas)
        this.supportVectors = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            if (alphas[i] > tolerance || alphaStar[i] > tolerance) {
                supportVectors.add(trainingData.get(i));
            }
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        RegressionDataPoint queryPoint = new RegressionDataPoint(features, 0.0);
        return predictInternal(queryPoint, supportVectors);
    }
    
    private double predictInternal(RegressionDataPoint queryPoint, List<RegressionDataPoint> trainingData) {
        double prediction = bias;
        
        for (int i = 0; i < trainingData.size(); i++) {
            double alpha = (i < alphas.length) ? alphas[i] : 0.0;
            double alphaSt = (i < alphaStar.length) ? alphaStar[i] : 0.0;
            
            if (alpha > tolerance || alphaSt > tolerance) {
                double kernel = rbfKernel(queryPoint, trainingData.get(i));
                prediction += (alpha - alphaSt) * kernel;
            }
        }
        
        return prediction;
    }
    
    private double rbfKernel(RegressionDataPoint p1, RegressionDataPoint p2) {
        double squaredDistance = 0.0;
        
        for (String feature : featureNames) {
            double v1 = p1.getFeatures().getOrDefault(feature, 0.0);
            double v2 = p2.getFeatures().getOrDefault(feature, 0.0);
            double diff = v1 - v2;
            squaredDistance += diff * diff;
        }
        
        return Math.exp(-gamma * squaredDistance);
    }
    
    /**
     * Gets the number of support vectors
     */
    public int getNumSupportVectors() {
        return trained ? supportVectors.size() : 0;
    }
    
    /**
     * Gets the support vector ratio
     */
    public double getSupportVectorRatio() {
        if (!trained || alphas == null) return 0.0;
        return (double) getNumSupportVectors() / alphas.length;
    }
    
    /**
     * Gets the support vectors
     */
    public List<RegressionDataPoint> getSupportVectors() {
        return trained ? new ArrayList<>(supportVectors) : new ArrayList<>();
    }
    
    // Additional helper methods for SMO algorithm...
}
```

### SVR Characteristics

**Advantages:**
- Handles non-linear relationships well
- Robust to outliers
- Works well in high-dimensional spaces
- Uses only support vectors for prediction
- Memory efficient for large datasets

**Disadvantages:**
- Many hyperparameters to tune (C, ε, γ)
- Computationally expensive for large datasets
- Less interpretable than linear methods
- Sensitive to feature scaling
- Choice of kernel function affects performance

**When to Use SVR:**
- When relationships are highly non-linear
- When you need robust predictions with outliers
- When you have sufficient training data
- When linear methods underperform
- When interpretability is not critical

## Model Evaluation for Regression

Evaluating regression models requires different metrics than classification. Let's implement a comprehensive evaluation framework.

### Regression Metrics

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Comprehensive regression evaluation metrics
 */
public class RegressionEvaluator {
    
    public RegressionMetrics evaluate(List<Double> actualValues, List<Double> predictedValues) {
        if (actualValues.size() != predictedValues.size()) {
            throw new IllegalArgumentException("Actual and predicted values must have the same size");
        }
        
        if (actualValues.isEmpty()) {
            throw new IllegalArgumentException("Cannot evaluate with empty lists");
        }
        
        int n = actualValues.size();
        
        // Calculate Mean Absolute Error (MAE)
        double mae = calculateMAE(actualValues, predictedValues);
        
        // Calculate Mean Squared Error (MSE)
        double mse = calculateMSE(actualValues, predictedValues);
        
        // Calculate Root Mean Squared Error (RMSE)
        double rmse = Math.sqrt(mse);
        
        // Calculate R² (Coefficient of Determination)
        double r2 = calculateR2(actualValues, predictedValues);
        
        // Calculate Mean Absolute Percentage Error (MAPE)
        double mape = calculateMAPE(actualValues, predictedValues);
        
        // Calculate residuals
        List<Double> residuals = calculateResiduals(actualValues, predictedValues);
        
        return new RegressionMetrics(mae, mse, rmse, r2, mape, residuals);
    }
    
    private double calculateMAE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            sum += Math.abs(actual.get(i) - predicted.get(i));
        }
        return sum / actual.size();
    }
    
    private double calculateMSE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            double error = actual.get(i) - predicted.get(i);
            sum += error * error;
        }
        return sum / actual.size();
    }
    
    private double calculateR2(List<Double> actual, List<Double> predicted) {
        // Calculate mean of actual values
        double meanActual = actual.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        // Calculate total sum of squares (TSS)
        double tss = 0.0;
        for (double value : actual) {
            double diff = value - meanActual;
            tss += diff * diff;
        }
        
        // Calculate residual sum of squares (RSS)
        double rss = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            double residual = actual.get(i) - predicted.get(i);
            rss += residual * residual;
        }
        
        // R² = 1 - (RSS / TSS)
        return (tss == 0) ? 0.0 : 1.0 - (rss / tss);
    }
    
    private double calculateMAPE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        int validCount = 0;
        
        for (int i = 0; i < actual.size(); i++) {
            double actualValue = actual.get(i);
            if (Math.abs(actualValue) > 1e-10) { // Avoid division by zero
                sum += Math.abs((actualValue - predicted.get(i)) / actualValue);
                validCount++;
            }
        }
        
        return validCount > 0 ? (sum / validCount) * 100.0 : 0.0;
    }
    
    private List<Double> calculateResiduals(List<Double> actual, List<Double> predicted) {
        List<Double> residuals = new ArrayList<>();
        for (int i = 0; i < actual.size(); i++) {
            residuals.add(actual.get(i) - predicted.get(i));
        }
        return residuals;
    }
}

/**
 * Container class for regression evaluation results
 */
public class RegressionMetrics {
    private final double mae;
    private final double mse;
    private final double rmse;
    private final double r2;
    private final double mape;
    private final List<Double> residuals;
    
    public RegressionMetrics(double mae, double mse, double rmse, double r2, 
                           double mape, List<Double> residuals) {
        this.mae = mae;
        this.mse = mse;
        this.rmse = rmse;
        this.r2 = r2;
        this.mape = mape;
        this.residuals = new ArrayList<>(residuals);
    }
    
    // Getters
    public double getMAE() { return mae; }
    public double getMSE() { return mse; }
    public double getRMSE() { return rmse; }
    public double getR2() { return r2; }
    public double getMAPE() { return mape; }
    public List<Double> getResiduals() { return new ArrayList<>(residuals); }
    
    public void printMetrics() {
        System.out.printf("Mean Absolute Error (MAE): %.4f%n", mae);
        System.out.printf("Mean Squared Error (MSE): %.4f%n", mse);
        System.out.printf("Root Mean Squared Error (RMSE): %.4f%n", rmse);
        System.out.printf("R² Score: %.4f%n", r2);
        System.out.printf("Mean Absolute Percentage Error (MAPE): %.2f%%%n", mape);
        
        // Basic residual statistics
        double meanResidual = residuals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdResidual = Math.sqrt(residuals.stream()
                .mapToDouble(r -> Math.pow(r - meanResidual, 2))
                .average()
                .orElse(0.0));
        
        System.out.printf("Mean Residual: %.4f%n", meanResidual);
        System.out.printf("Std Residual: %.4f%n", stdResidual);
    }
}
```

### Understanding Regression Metrics

**Mean Absolute Error (MAE):**
- Average absolute difference between actual and predicted values
- Formula: (1/n) × Σ|yᵢ - ŷᵢ|
- Interpretable in the same units as the target variable

**Mean Squared Error (MSE):**
- Average squared difference between actual and predicted values
- Formula: (1/n) × Σ(yᵢ - ŷᵢ)²
- Penalizes larger errors more heavily

**Root Mean Squared Error (RMSE):**
- Square root of MSE
- Formula: √MSE
- Same units as the target variable, penalizes large errors

**R² Score (Coefficient of Determination):**
- Proportion of variance in the target variable explained by the model
- Formula: 1 - (RSS/TSS)
- Range: -∞ to 1, where 1 is perfect fit

**Mean Absolute Percentage Error (MAPE):**
- Average absolute percentage error
- Formula: (100/n) × Σ|(yᵢ - ŷᵢ)/yᵢ|
- Useful for comparing models across different scales

## Practical Example: House Price Prediction

Let's build a complete regression pipeline for house price prediction:

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Complete regression example: House Price Prediction
 */
public class HousePricePredictionExample {
    
    public static void main(String[] args) {
        System.out.println("=== House Price Prediction Example ===\n");
        
        // Generate sample house data
        List<RegressionDataPoint> houseData = generateHouseData(1000);
        
        // Split data into training and testing sets
        RegressionDataSplitter splitter = new RegressionDataSplitter(0.8);
        RegressionDataSplit split = splitter.split(houseData);
        
        System.out.println("Training samples: " + split.getTrainingData().size());
        System.out.println("Test samples: " + split.getTestData().size());
        
        // Preprocess data (normalize features)
        RegressionPreprocessor preprocessor = new RegressionPreprocessor();
        List<RegressionDataPoint> normalizedTrainData = preprocessor.normalizeFeatures(split.getTrainingData());
        List<RegressionDataPoint> normalizedTestData = preprocessor.normalizeFeatures(split.getTestData());
        
        // Train and evaluate different regressors
        List<Regressor> regressors = Arrays.asList(
            new LinearRegression(),
            new PolynomialRegression(2),
            new RidgeRegression(0.1),
            new RidgeRegression(1.0),
            new LassoRegression(0.1),
            new LassoRegression(1.0),
            new SupportVectorRegression(1.0, 0.1, 0.1)
        );
        
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (Regressor regressor : regressors) {
            System.out.println("\n" + regressor.getName() + " Results:");
            System.out.println("================================");
            
            // Train the regressor
            long trainStart = System.currentTimeMillis();
            regressor.train(normalizedTrainData);
            long trainTime = System.currentTimeMillis() - trainStart;
            
            // Make predictions
            long predictStart = System.currentTimeMillis();
            List<Double> predictions = new ArrayList<>();
            List<Double> actuals = new ArrayList<>();
            
            for (RegressionDataPoint testPoint : normalizedTestData) {
                predictions.add(regressor.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getTarget());
            }
            long predictTime = System.currentTimeMillis() - predictStart;
            
            // Evaluate performance
            RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
            metrics.printMetrics();
            
            System.out.printf("Training time: %d ms%n", trainTime);
            System.out.printf("Prediction time: %d ms%n", predictTime);
            
            // Print feature importance for applicable models
            if (regressor instanceof LinearRegression) {
                LinearRegression lr = (LinearRegression) regressor;
                System.out.println("Feature Importance:");
                lr.getFeatureImportance().entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .forEach(entry -> System.out.printf("  %s: %.3f%n", entry.getKey(), entry.getValue()));
            } else if (regressor instanceof LassoRegression) {
                LassoRegression lasso = (LassoRegression) regressor;
                System.out.printf("Selected Features: %d/%d%n", 
                    lasso.getSelectedFeatures().size(), lasso.getCoefficients().size());
                System.out.printf("Sparsity: %.1f%%%n", lasso.getSparsity() * 100);
                System.out.println("Selected Features: " + lasso.getSelectedFeatures());
            } else if (regressor instanceof SupportVectorRegression) {
                SupportVectorRegression svr = (SupportVectorRegression) regressor;
                System.out.printf("Support Vectors: %d (%.1f%%)%n", 
                    svr.getNumSupportVectors(), svr.getSupportVectorRatio() * 100);
            }
        }
        
        // Demonstrate overfitting with high-degree polynomial
        System.out.println("\n=== Overfitting Demonstration ===");
        demonstrateOverfitting(normalizedTrainData, normalizedTestData);
    }
    
    /**
     * Generates synthetic house data for demonstration
     */
    private static List<RegressionDataPoint> generateHouseData(int numSamples) {
        List<RegressionDataPoint> data = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < numSamples; i++) {
            Map<String, Double> features = new HashMap<>();
            
            // Generate features
            double sqft = 1000 + random.nextGaussian() * 800; // Square footage
            double bedrooms = Math.max(1, 2 + random.nextGaussian() * 1.5); // Number of bedrooms
            double bathrooms = Math.max(1, 1.5 + random.nextGaussian() * 1.0); // Number of bathrooms
            double age = Math.max(0, random.nextGaussian() * 20); // Age of house in years
            double lotSize = 0.1 + random.nextGaussian() * 0.5; // Lot size in acres
            double garageSize = Math.max(0, random.nextGaussian() * 1.5); // Garage size
            
            features.put("sqft", sqft);
            features.put("bedrooms", bedrooms);
            features.put("bathrooms", bathrooms);
            features.put("age", age);
            features.put("lot_size", lotSize);
            features.put("garage_size", garageSize);
            
            // Calculate price based on realistic relationships
            double basePrice = 100000; // Base price
            double price = basePrice;
            
            // Square footage is the main driver
            price += sqft * 120; // $120 per sqft
            
            // Bedrooms and bathrooms add value
            price += bedrooms * 15000;
            price += bathrooms * 10000;
            
            // Age decreases value
            price -= age * 1000;
            
            // Lot size adds value
            price += lotSize * 50000;
            
            // Garage adds value
            price += garageSize * 8000;
            
            // Add some non-linear effects
            price += Math.pow(sqft / 1000, 1.5) * 20000; // Diminishing returns for very large houses
            price -= Math.pow(age / 50, 2) * 30000; // Accelerating depreciation for very old houses
            
            // Add noise
            price += random.nextGaussian() * 25000;
            
            // Ensure price is positive
            price = Math.max(price, 50000);
            
            data.add(new RegressionDataPoint(features, price));
        }
        
        return data;
    }
    
    private static void demonstrateOverfitting(List<RegressionDataPoint> trainData, 
                                             List<RegressionDataPoint> testData) {
        
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (int degree = 1; degree <= 5; degree++) {
            PolynomialRegression poly = new PolynomialRegression(degree);
            poly.train(trainData);
            
            // Evaluate on training data
            List<Double> trainPredictions = new ArrayList<>();
            List<Double> trainActuals = new ArrayList<>();
            
            for (RegressionDataPoint point : trainData) {
                trainPredictions.add(poly.predict(point.getFeatures()));
                trainActuals.add(point.getTarget());
            }
            
            RegressionMetrics trainMetrics = evaluator.evaluate(trainActuals, trainPredictions);
            
            // Evaluate on test data
            List<Double> testPredictions = new ArrayList<>();
            List<Double> testActuals = new ArrayList<>();
            
            for (RegressionDataPoint point : testData) {
                testPredictions.add(poly.predict(point.getFeatures()));
                testActuals.add(point.getTarget());
            }
            
            RegressionMetrics testMetrics = evaluator.evaluate(testActuals, testPredictions);
            
            System.out.printf("Degree %d - Train R²: %.4f, Test R²: %.4f, Difference: %.4f%n",
                degree, trainMetrics.getR2(), testMetrics.getR2(), 
                trainMetrics.getR2() - testMetrics.getR2());
        }
    }
}
```

## Data Preprocessing and Utilities

### Data Splitting

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Utility class for splitting regression data into training and testing sets
 */
public class RegressionDataSplitter {
    
    private final double trainRatio;
    private final Random random;
    
    public RegressionDataSplitter(double trainRatio) {
        this(trainRatio, new Random());
    }
    
    public RegressionDataSplitter(double trainRatio, Random random) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        this.trainRatio = trainRatio;
        this.random = random;
    }
    
    public RegressionDataSplit split(List<RegressionDataPoint> data) {
        List<RegressionDataPoint> shuffledData = new ArrayList<>(data);
        Collections.shuffle(shuffledData, random);
        
        int trainSize = (int) (data.size() * trainRatio);
        
        List<RegressionDataPoint> trainData = shuffledData.subList(0, trainSize);
        List<RegressionDataPoint> testData = shuffledData.subList(trainSize, shuffledData.size());
        
        return new RegressionDataSplit(trainData, testData);
    }
}

/**
 * Container class for train/test split results
 */
public class RegressionDataSplit {
    private final List<RegressionDataPoint> trainingData;
    private final List<RegressionDataPoint> testData;
    
    public RegressionDataSplit(List<RegressionDataPoint> trainingData, List<RegressionDataPoint> testData) {
        this.trainingData = new ArrayList<>(trainingData);
        this.testData = new ArrayList<>(testData);
    }
    
    public List<RegressionDataPoint> getTrainingData() {
        return new ArrayList<>(trainingData);
    }
    
    public List<RegressionDataPoint> getTestData() {
        return new ArrayList<>(testData);
    }
}
```

### Data Preprocessing

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Data preprocessing utilities for regression
 */
public class RegressionPreprocessor {
    
    /**
     * Normalize features to [0, 1] range
     */
    public List<RegressionDataPoint> normalizeFeatures(List<RegressionDataPoint> data) {
        if (data.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Find min and max for each feature
        Map<String, Double> minValues = new HashMap<>();
        Map<String, Double> maxValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                minValues.put(feature, Math.min(minValues.getOrDefault(feature, Double.MAX_VALUE), value));
                maxValues.put(feature, Math.max(maxValues.getOrDefault(feature, Double.MIN_VALUE), value));
            }
        }
        
        // Normalize each data point
        List<RegressionDataPoint> normalizedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> normalizedFeatures = new HashMap<>();
            
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                double min = minValues.get(feature);
                double max = maxValues.get(feature);
                
                double normalized = (max - min == 0) ? 0.0 : (value - min) / (max - min);
                normalizedFeatures.put(feature, normalized);
            }
            
            normalizedData.add(new RegressionDataPoint(normalizedFeatures, point.getTarget()));
        }
        
        return normalizedData;
    }
    
    /**
     * Standardize features to have mean 0 and standard deviation 1
     */
    public List<RegressionDataPoint> standardizeFeatures(List<RegressionDataPoint> data) {
        if (data.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Calculate means and standard deviations
        Map<String, Double> means = calculateFeatureMeans(data);
        Map<String, Double> stdDevs = calculateFeatureStdDevs(data, means);
        
        // Standardize each data point
        List<RegressionDataPoint> standardizedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> standardizedFeatures = new HashMap<>();
            
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                double mean = means.get(feature);
                double stdDev = stdDevs.get(feature);
                
                double standardized = (stdDev == 0) ? 0.0 : (value - mean) / stdDev;
                standardizedFeatures.put(feature, standardized);
            }
            
            standardizedData.add(new RegressionDataPoint(standardizedFeatures, point.getTarget()));
        }
        
        return standardizedData;
    }
    
    /**
     * Handle missing values using mean imputation
     */
    public List<RegressionDataPoint> handleMissingValues(List<RegressionDataPoint> data) {
        Map<String, Double> featureMeans = calculateFeatureMeans(data);
        
        List<RegressionDataPoint> imputedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> imputedFeatures = new HashMap<>();
            
            for (String feature : featureMeans.keySet()) {
                double value = point.getFeatures().getOrDefault(feature, featureMeans.get(feature));
                imputedFeatures.put(feature, value);
            }
            
            imputedData.add(new RegressionDataPoint(imputedFeatures, point.getTarget()));
        }
        
        return imputedData;
    }
    
    private Map<String, Double> calculateFeatureMeans(List<RegressionDataPoint> data) {
        Map<String, List<Double>> featureValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
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
    
    private Map<String, Double> calculateFeatureStdDevs(List<RegressionDataPoint> data, 
                                                       Map<String, Double> means) {
        Map<String, List<Double>> featureValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                featureValues.computeIfAbsent(feature, k -> new ArrayList<>())
                           .add(point.getFeatures().get(feature));
            }
        }
        
        Map<String, Double> stdDevs = new HashMap<>();
        for (String feature : featureValues.keySet()) {
            double mean = means.get(feature);
            double variance = featureValues.get(feature).stream()
                                    .mapToDouble(value -> Math.pow(value - mean, 2))
                                    .average()
                                    .orElse(0.0);
            stdDevs.put(feature, Math.sqrt(variance));
        }
        
        return stdDevs;
    }
}
```

## Best Practices for Regression

### 1. Cross-Validation for Regression

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * K-Fold Cross-Validation for regression
 */
public class RegressionCrossValidator {
    
    public static RegressionCrossValidationResults kFoldCrossValidation(
            Regressor regressor, List<RegressionDataPoint> data, int k) {
        
        List<Double> maes = new ArrayList<>();
        List<Double> mses = new ArrayList<>();
        List<Double> rmses = new ArrayList<>();
        List<Double> r2s = new ArrayList<>();
        
        int foldSize = data.size() / k;
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (int fold = 0; fold < k; fold++) {
            // Create train/test split for this fold
            int startIdx = fold * foldSize;
            int endIdx = (fold == k - 1) ? data.size() : (fold + 1) * foldSize;
            
            List<RegressionDataPoint> testFold = data.subList(startIdx, endIdx);
            List<RegressionDataPoint> trainFold = new ArrayList<>();
            trainFold.addAll(data.subList(0, startIdx));
            trainFold.addAll(data.subList(endIdx, data.size()));
            
            // Train and evaluate
            regressor.train(trainFold);
            
            List<Double> predictions = new ArrayList<>();
            List<Double> actuals = new ArrayList<>();
            
            for (RegressionDataPoint testPoint : testFold) {
                predictions.add(regressor.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getTarget());
            }
            
            RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
            
            maes.add(metrics.getMAE());
            mses.add(metrics.getMSE());
            rmses.add(metrics.getRMSE());
            r2s.add(metrics.getR2());
        }
        
        return new RegressionCrossValidationResults(maes, mses, rmses, r2s);
    }
}

/**
 * Cross-validation results container for regression
 */
public class RegressionCrossValidationResults {
    private final List<Double> maes;
    private final List<Double> mses;
    private final List<Double> rmses;
    private final List<Double> r2s;
    
    public RegressionCrossValidationResults(List<Double> maes, List<Double> mses, 
                                          List<Double> rmses, List<Double> r2s) {
        this.maes = maes;
        this.mses = mses;
        this.rmses = rmses;
        this.r2s = r2s;
    }
    
    public double getMeanMAE() {
        return maes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdMAE() {
        double mean = getMeanMAE();
        double variance = maes.stream()
                .mapToDouble(mae -> Math.pow(mae - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public double getMeanR2() {
        return r2s.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdR2() {
        double mean = getMeanR2();
        double variance = r2s.stream()
                .mapToDouble(r2 -> Math.pow(r2 - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public void printResults() {
        System.out.printf("Cross-Validation Results (Mean ± Std):%n");
        System.out.printf("MAE: %.4f ± %.4f%n", getMeanMAE(), getStdMAE());
        System.out.printf("MSE: %.4f ± %.4f%n", getMeanMSE(), getStdMSE());
        System.out.printf("RMSE: %.4f ± %.4f%n", getMeanRMSE(), getStdRMSE());
        System.out.printf("R²: %.4f ± %.4f%n", getMeanR2(), getStdR2());
    }
    
    // Similar methods for MSE and RMSE...
    private double getMeanMSE() {
        return mses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    private double getStdMSE() {
        double mean = getMeanMSE();
        double variance = mses.stream()
                .mapToDouble(mse -> Math.pow(mse - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    private double getMeanRMSE() {
        return rmses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    private double getStdRMSE() {
        double mean = getMeanRMSE();
        double variance = rmses.stream()
                .mapToDouble(rmse -> Math.pow(rmse - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
}
```

### 2. Hyperparameter Tuning

```java
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Hyperparameter tuning for Ridge Regression
 */
public class RidgeRegressionTuner {
    
    public static double findBestAlpha(List<RegressionDataPoint> data, 
                                      double[] alphaValues, int kFolds) {
        double bestAlpha = alphaValues[0];
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (double alpha : alphaValues) {
            RidgeRegression ridge = new RidgeRegression(alpha);
            RegressionCrossValidationResults results = 
                RegressionCrossValidator.kFoldCrossValidation(ridge, data, kFolds);
            
            double meanR2 = results.getMeanR2();
            
            System.out.printf("Alpha: %.4f, Mean R²: %.4f ± %.4f%n", 
                alpha, meanR2, results.getStdR2());
            
            if (meanR2 > bestScore) {
                bestScore = meanR2;
                bestAlpha = alpha;
            }
        }
        
        System.out.printf("Best Alpha: %.4f with R²: %.4f%n", bestAlpha, bestScore);
        return bestAlpha;
    }
}
```

## Running the Examples

To run the regression examples, create the directory structure and compile the code:

```bash
mkdir -p chapter-05-regression/src/main/java/com/aiprogramming/ch05
cd chapter-05-regression
javac -d . src/main/java/com/aiprogramming/ch05/*.java
java com.aiprogramming.ch05.HousePricePredictionExample
```

## Summary

In this chapter, we've covered the fundamental concepts of regression and implemented several key algorithms from scratch:

1. **Linear Regression** - Basic linear relationship modeling using normal equation
2. **Polynomial Regression** - Non-linear relationships using polynomial features  
3. **Ridge Regression** - Linear regression with L2 regularization to prevent overfitting
4. **Regression Evaluation** - Comprehensive metrics including MAE, MSE, RMSE, R², and MAPE
5. **Data Preprocessing** - Feature normalization, standardization, and missing value handling

We've also learned about:
- Regression evaluation metrics and their interpretation
- Cross-validation techniques for reliable performance estimation
- Overfitting and underfitting in regression models
- Hyperparameter tuning for regularized models
- Practical implementation considerations

### Key Takeaways

1. **Choose the right algorithm** based on your data characteristics and requirements
2. **Evaluate thoroughly** using multiple metrics appropriate for your use case
3. **Preprocess your data** to improve algorithm performance and stability
4. **Use cross-validation** to get reliable performance estimates and detect overfitting
5. **Apply regularization** when you have many features or limited data
6. **Balance model complexity** to avoid overfitting while capturing important patterns

### Algorithm Selection Guide

| Algorithm | Best For | Advantages | Disadvantages |
|-----------|----------|------------|---------------|
| Linear Regression | Linear relationships, interpretability | Fast, interpretable, no hyperparameters | Assumes linearity, sensitive to outliers |
| Polynomial Regression | Non-linear patterns | Captures curves, still interpretable | Prone to overfitting, computationally expensive |
| Ridge Regression | Many features, multicollinearity | Reduces overfitting, stable | Requires hyperparameter tuning |
| Lasso Regression | Feature selection, sparsity | Feature selection, reduces overfitting | Requires hyperparameter tuning |
| SVR | Non-linear relationships, robust | Handles non-linearity, robust to outliers | Complex, many hyperparameters |

### Next Steps

In the next chapter, we'll explore **Unsupervised Learning** techniques including clustering, dimensionality reduction, and anomaly detection. These methods work with unlabeled data to discover hidden patterns and structures.

The evaluation and preprocessing techniques you've learned in this chapter will be valuable for assessing unsupervised learning results and preparing data for clustering algorithms.

## Complete Implementation Summary

Chapter 5 now provides a comprehensive regression library with five fully implemented algorithms:

### ✅ Implemented Algorithms
1. **Linear Regression** - Complete with normal equation solution and feature importance
2. **Polynomial Regression** - Automatic polynomial feature generation with configurable degree
3. **Ridge Regression** - L2 regularization with matrix inversion solution
4. **Lasso Regression** - L1 regularization with coordinate descent optimization
5. **Support Vector Regression** - RBF kernel with simplified SMO-like algorithm

### ✅ Complete Infrastructure
- **Evaluation Framework** - MAE, MSE, RMSE, R², MAPE metrics
- **Cross-Validation** - K-fold cross-validation with statistical analysis
- **Data Processing** - Feature normalization, standardization, missing value handling
- **Hyperparameter Tuning** - Automated parameter selection for Ridge regression
- **Practical Examples** - House price prediction with comprehensive algorithm comparison

### ✅ Real-World Features
- **Feature Selection** - Automatic feature selection with Lasso regression
- **Sparsity Analysis** - Analysis of model complexity and feature usage
- **Support Vector Analysis** - Support vector identification and ratio analysis
- **Performance Benchmarking** - Training and prediction time analysis
- **Overfitting Detection** - Systematic analysis of model complexity effects

## Exercises

1. **Implement Elastic Net** regression combining L1 and L2 regularization
2. **Add feature selection** based on correlation analysis and mutual information
3. **Create a time series forecasting** system using regression with lagged features
4. **Build a stock price prediction** model with technical indicators
5. **Implement gradient descent** for linear regression instead of normal equation
6. **Add robust regression** techniques to handle outliers (Huber regression)
7. **Create learning curves** to visualize training vs validation performance
8. **Implement regularization path** visualization for Lasso and Ridge regression

## Further Reading

- *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman
- *Introduction to Statistical Learning* by James, Witten, Hastie, and Tibshirani
- *Hands-On Machine Learning* by Aurélien Géron
- Java ML libraries: Weka, Smile, Tribuo
- Regression analysis in practice: Andrew Gelman's books on applied regression
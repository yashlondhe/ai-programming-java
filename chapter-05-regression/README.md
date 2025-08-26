# Chapter 5: Regression Algorithms

This chapter implements various regression algorithms in Java, demonstrating how to predict continuous target values from input features.

## 🎯 Learning Objectives

- Understand different regression algorithms and their use cases
- Implement Linear, Polynomial, Ridge, Lasso, and Support Vector Regression
- Learn about regularization techniques and their impact on model performance
- Explore feature selection and importance analysis
- Understand overfitting and how to detect it
- Master evaluation metrics for regression problems

## 🚀 Quick Start

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Building and Running

```bash
# Compile the project
mvn clean compile

# Run the main demo
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch05.HousePricePredictionExample"

# Run tests
mvn test

# Create executable JAR
mvn package
```

## 📊 Algorithms Implemented

### 1. Linear Regression
- **File**: `LinearRegression.java`
- **Method**: Normal equation with QR decomposition
- **Features**: Robust matrix operations using Apache Commons Math
- **Use Case**: Simple linear relationships between features and target

### 2. Polynomial Regression
- **File**: `PolynomialRegression.java`
- **Method**: Feature transformation + Linear regression
- **Features**: Configurable polynomial degree
- **Use Case**: Non-linear relationships

### 3. Ridge Regression
- **File**: `RidgeRegression.java`
- **Method**: L2 regularization
- **Features**: Configurable alpha parameter
- **Use Case**: Preventing overfitting, handling multicollinearity

### 4. Lasso Regression
- **File**: `LassoRegression.java`
- **Method**: L1 regularization with coordinate descent
- **Features**: Feature selection, sparsity
- **Use Case**: Feature selection, handling high-dimensional data

### 5. Support Vector Regression
- **File**: `SupportVectorRegression.java`
- **Method**: SMO algorithm with kernel trick
- **Features**: Configurable C, epsilon, and gamma parameters
- **Use Case**: Non-linear regression with support vectors

## 🛠️ Utility Classes

### Data Management
- **`RegressionDataPoint`**: Represents a single data point with features and target
- **`Regressor`**: Interface for all regression algorithms
- **`RegressionDataSplitter`**: Splits data into training and test sets
- **`RegressionDataSplit`**: Container for training and test data

### Preprocessing
- **`RegressionPreprocessor`**: Feature normalization and standardization
- **`RidgeRegressionTuner`**: Hyperparameter tuning for Ridge regression

### Evaluation
- **`RegressionEvaluator`**: Calculates comprehensive evaluation metrics
- **`RegressionMetrics`**: Container for evaluation results
- **`RegressionCrossValidator`**: K-fold cross-validation
- **`RegressionCrossValidationResults`**: Cross-validation results

## 📈 Evaluation Metrics

The framework calculates the following metrics:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **MSE** (Mean Squared Error): Average squared difference between predictions and actual values
- **RMSE** (Root Mean Squared Error): Square root of MSE, in same units as target
- **R²** (Coefficient of Determination): Proportion of variance explained by the model
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **Residuals**: Distribution of prediction errors

## 🏠 House Price Prediction Example

The main demo (`HousePricePredictionExample.java`) demonstrates:

1. **Data Generation**: Synthetic house price data with realistic relationships
2. **Data Splitting**: 80% training, 20% test split
3. **Feature Normalization**: Scaling features to [0,1] range
4. **Model Training**: All regression algorithms trained on the same data
5. **Performance Comparison**: Side-by-side evaluation of all algorithms
6. **Overfitting Demonstration**: Polynomial regression with increasing degrees

### Sample Output
```
LinearRegression Results:
================================
Mean Absolute Error (MAE): 75588.4797
Mean Squared Error (MSE): 7855066994.6219
Root Mean Squared Error (RMSE): 88628.8158
R² Score: 0.4288
Mean Absolute Percentage Error (MAPE): 28.79%
```

## 🔧 Code Examples

### Basic Usage
```java
// Create and train a linear regression model
LinearRegression lr = new LinearRegression();
lr.train(trainingData);

// Make predictions
Map<String, Double> features = new HashMap<>();
features.put("sqft", 2000.0);
features.put("bedrooms", 3.0);
double prediction = lr.predict(features);

// Get model coefficients
Map<String, Double> coefficients = lr.getCoefficients();
double intercept = lr.getIntercept();
```

### Ridge Regression with Tuning
```java
// Find optimal alpha using cross-validation
double bestAlpha = RidgeRegressionTuner.findBestAlpha(data, 5);

// Train model with optimal parameters
RidgeRegression ridge = new RidgeRegression(bestAlpha);
ridge.train(trainingData);
```

### Feature Selection with Lasso
```java
LassoRegression lasso = new LassoRegression(0.1);
lasso.train(trainingData);

// Get selected features
List<String> selectedFeatures = lasso.getSelectedFeatures();
double sparsity = lasso.getSparsity();
```

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=LinearRegressionTest

# Run with detailed output
mvn test -Dtest=*Test -Dsurefire.useFile=false
```

## 📚 Key Features

### Robust Implementation
- **Apache Commons Math**: Professional-grade matrix operations
- **Error Handling**: Graceful fallback for numerical issues
- **Input Validation**: Comprehensive parameter checking

### Performance Optimization
- **Efficient Algorithms**: Optimized implementations for each method
- **Memory Management**: Minimal memory footprint
- **Fast Training**: Quick convergence for most datasets

### Extensibility
- **Interface Design**: Easy to add new regression algorithms
- **Modular Structure**: Independent components for different functionalities
- **Configurable Parameters**: Tunable hyperparameters for each algorithm

## 🎓 Educational Value

This implementation demonstrates:

1. **Algorithm Understanding**: Clear, readable implementations of complex algorithms
2. **Software Engineering**: Proper Java practices, error handling, and testing
3. **Machine Learning Concepts**: Regularization, feature selection, overfitting
4. **Performance Analysis**: Comprehensive evaluation and comparison
5. **Real-world Application**: Practical house price prediction example

## 🔍 Algorithm Performance Comparison

Based on the house price prediction example:

| Algorithm | R² Score | MAE | Training Time | Use Case |
|-----------|----------|-----|---------------|----------|
| Linear Regression | 0.4288 | 75,588 | 30ms | Baseline, simple relationships |
| Ridge Regression (α=1.0) | 0.5207 | 70,225 | 4ms | Regularization, multicollinearity |
| Lasso Regression | 0.4288 | 75,588 | 12ms | Feature selection |
| Polynomial Regression | 0.2688 | 80,189 | 82ms | Non-linear relationships |
| SVR | -6.1548 | 290,920 | 455ms | Non-linear, needs tuning |

## 🚀 Next Steps

1. **Parameter Tuning**: Implement grid search for hyperparameter optimization
2. **Additional Algorithms**: Add Random Forest, Neural Networks, or XGBoost
3. **Real Data**: Use actual housing datasets (Boston, California, etc.)
4. **Advanced Features**: Implement feature engineering and selection techniques
5. **Production Ready**: Add model persistence, API endpoints, and monitoring

## 📖 Chapter Summary

Chapter 5 provides a comprehensive introduction to regression algorithms, covering:

- **Fundamental Algorithms**: Linear, Polynomial, Ridge, Lasso, SVR
- **Regularization Techniques**: L1 and L2 regularization for preventing overfitting
- **Evaluation Framework**: Comprehensive metrics for model assessment
- **Practical Application**: Real-world house price prediction example
- **Overfitting Analysis**: Demonstration of model complexity vs. generalization

The implementations are production-ready, well-tested, and serve as excellent educational resources for understanding regression algorithms in practice.
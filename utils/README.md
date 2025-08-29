# AI Programming Utils

A comprehensive utility library for AI/ML applications in Java. This module provides common functionality needed across multiple chapters of the AI Programming with Java book.

## 📦 Contents

### DataUtils
Common data processing utilities for AI/ML applications:
- **CSV Loading**: Load and parse CSV files
- **Data Conversion**: Convert string data to numeric arrays
- **Normalization**: Min-max scaling and z-score standardization
- **Train-Test Splitting**: Split data into training and testing sets
- **Evaluation Metrics**: Accuracy, MSE, RMSE, R-squared for model evaluation
- **Debugging**: Matrix and vector printing utilities

### MatrixUtils
Matrix operations and mathematical functions:
- **Matrix Creation**: Zeros, ones, identity, random matrices
- **Matrix Operations**: Transpose, multiplication, addition, subtraction
- **Element-wise Operations**: Hadamard product, function application
- **Vector Operations**: Dot product, norm, normalization
- **Common Functions**: Sigmoid, tanh, ReLU, exponential, logarithmic
- **Utility Functions**: Copy, equality checking, statistics

### ValidationUtils
Input validation and error checking:
- **Matrix Validation**: Dimension checking, null validation, finite value checking
- **Vector Validation**: Length validation, null checking
- **Value Validation**: Range checking, probability validation, positive/non-negative
- **Statistical Validation**: Symmetric matrix, positive definite matrix
- **General Validation**: String validation, collection validation

### StatisticsUtils
Statistical operations and probability distributions:
- **Descriptive Statistics**: Mean, median, mode, variance, standard deviation
- **Distribution Analysis**: Skewness, kurtosis, percentiles, IQR
- **Correlation Analysis**: Correlation coefficient, covariance, covariance matrix
- **Outlier Detection**: IQR-based outlier detection, z-scores
- **Information Theory**: Entropy, cross-entropy, KL divergence
- **Random Generation**: Normal and uniform distribution sampling

## 🚀 Quick Start

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Installation
```bash
cd utils
mvn clean install
```

### Usage Example
```java
import com.aiprogramming.utils.*;

// Load and process data
List<String[]> rawData = DataUtils.loadCSV("data.csv", true);
DataUtils.Pair<double[][], double[]> data = DataUtils.convertToNumeric(
    rawData, new int[]{0, 1, 2}, 3);

// Normalize features
double[][] normalizedFeatures = DataUtils.normalize(data.first);

// Split data
DataUtils.TrainTestSplit split = DataUtils.trainTestSplit(
    normalizedFeatures, data.second, 0.2, 42L);

// Create random weights for neural network
double[][] weights = MatrixUtils.randomNormal(10, 5, 0.0, 0.1, 42L);

// Apply activation function
double[][] activated = MatrixUtils.apply(weights, MatrixUtils.Functions.SIGMOID);

// Calculate statistics
double mean = StatisticsUtils.mean(data.second);
double std = StatisticsUtils.standardDeviation(data.second);
```

## 📚 API Documentation

### DataUtils
```java
// Data loading
List<String[]> loadCSV(String filePath, boolean hasHeader)

// Data conversion
Pair<double[][], double[]> convertToNumeric(List<String[]> data, 
                                          int[] featureColumns, int targetColumn)

// Data preprocessing
double[][] normalize(double[][] data)
double[][] standardize(double[][] data)
TrainTestSplit trainTestSplit(double[][] X, double[] y, 
                             double testSize, long randomSeed)

// Evaluation metrics
double accuracy(double[] yTrue, double[] yPred)
double meanSquaredError(double[] yTrue, double[] yPred)
double rSquared(double[] yTrue, double[] yPred)
```

### MatrixUtils
```java
// Matrix creation
double[][] zeros(int rows, int cols)
double[][] ones(int rows, int cols)
double[][] identity(int size)
double[][] random(int rows, int cols, double min, double max, long seed)
double[][] randomNormal(int rows, int cols, double mean, double std, long seed)

// Matrix operations
double[][] transpose(double[][] matrix)
double[][] multiply(double[][] A, double[][] B)
double[][] add(double[][] A, double[][] B)
double[][] subtract(double[][] A, double[][] B)

// Vector operations
double dotProduct(double[] a, double[] b)
double norm(double[] vector)
double[] normalize(double[] vector)

// Function application
double[][] apply(double[][] matrix, MatrixFunction function)
```

### ValidationUtils
```java
// Matrix validation
void validateMatrix(double[][] matrix, String name)
void validateMatrixDimensions(double[][] matrix, int rows, int cols, String name)
void validateFiniteValues(double[][] matrix, String name)

// Vector validation
void validateVector(double[] vector, String name)
void validateVectorLength(double[] vector, int length, String name)

// Value validation
void validateRange(double value, double min, double max, String name)
void validateProbability(double value, String name)
void validatePositive(double value, String name)
```

### StatisticsUtils
```java
// Descriptive statistics
double mean(double[] data)
double median(double[] data)
double variance(double[] data)
double standardDeviation(double[] data)

// Distribution analysis
double skewness(double[] data)
double kurtosis(double[] data)
double[] percentiles(double[] data, double[] percentiles)

// Correlation and covariance
double correlation(double[] x, double[] y)
double covariance(double[] x, double[] y)
double[][] covarianceMatrix(double[][] data)

// Outlier detection
boolean[] detectOutliers(double[] data)
double[] zScores(double[] data)

// Information theory
double entropy(double[] probabilities)
double crossEntropy(double[] p, double[] q)
double klDivergence(double[] p, double[] q)
```

## 🧪 Testing

Run the test suite:
```bash
mvn test
```

Run tests with coverage:
```bash
mvn test -Pcoverage
```

## 📦 Building

Build the JAR file:
```bash
mvn clean package
```

Build with dependencies (fat JAR):
```bash
mvn clean package assembly:single
```

Generate Javadoc:
```bash
mvn javadoc:javadoc
```

## 🔧 Integration

### As a Maven Dependency
Add to your `pom.xml`:
```xml
<dependency>
    <groupId>com.aiprogramming</groupId>
    <artifactId>utils</artifactId>
    <version>1.0.0</version>
</dependency>
```

### As a JAR File
Include the JAR file in your project's classpath:
```bash
java -cp utils-1.0.0.jar your.MainClass
```

## 📖 Examples

### Data Preprocessing Pipeline
```java
// Load data
List<String[]> rawData = DataUtils.loadCSV("iris.csv", true);

// Convert to numeric
DataUtils.Pair<double[][], double[]> data = DataUtils.convertToNumeric(
    rawData, new int[]{0, 1, 2, 3}, 4);

// Validate data
ValidationUtils.validateFiniteValues(data.first, "features");
ValidationUtils.validateFiniteValues(data.second, "targets");

// Preprocess
double[][] normalized = DataUtils.normalize(data.first);
double[][] standardized = DataUtils.standardize(data.first);

// Split data
DataUtils.TrainTestSplit split = DataUtils.trainTestSplit(
    standardized, data.second, 0.3, 42L);
```

### Neural Network Utilities
```java
// Initialize weights
double[][] weights1 = MatrixUtils.randomNormal(10, 5, 0.0, 0.1, 42L);
double[][] weights2 = MatrixUtils.randomNormal(5, 1, 0.0, 0.1, 43L);

// Forward pass
double[][] layer1 = MatrixUtils.multiply(input, weights1);
double[][] activated1 = MatrixUtils.apply(layer1, MatrixUtils.Functions.RELU);
double[][] layer2 = MatrixUtils.multiply(activated1, weights2);
double[][] output = MatrixUtils.apply(layer2, MatrixUtils.Functions.SIGMOID);
```

### Statistical Analysis
```java
// Calculate descriptive statistics
double mean = StatisticsUtils.mean(data);
double std = StatisticsUtils.standardDeviation(data);
double skew = StatisticsUtils.skewness(data);
double kurt = StatisticsUtils.kurtosis(data);

// Detect outliers
boolean[] outliers = StatisticsUtils.detectOutliers(data);
int outlierCount = 0;
for (boolean outlier : outliers) {
    if (outlier) outlierCount++;
}

// Calculate correlation
double correlation = StatisticsUtils.correlation(feature1, feature2);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the "AI Programming with Java" book project
- Designed to be educational and practical
- Focused on clarity and ease of use
- Comprehensive test coverage for reliability

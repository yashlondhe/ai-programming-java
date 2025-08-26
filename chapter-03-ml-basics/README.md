# Chapter 3: Introduction to Machine Learning

This chapter introduces the fundamental concepts of machine learning and demonstrates how to implement them using Java. The chapter covers the complete machine learning workflow from data preprocessing to model deployment.

## Learning Objectives

- Understand the core concepts and principles of machine learning
- Learn about different types of ML problems and their applications
- Master the ML workflow from data to prediction
- Implement data preprocessing and feature engineering techniques
- Understand model evaluation metrics and validation strategies
- Learn to identify and handle overfitting and underfitting
- Build practical ML applications using Java

## Chapter Content

### 1. Machine Learning Fundamentals
- **What is Machine Learning?**: Understanding the difference between traditional programming and ML
- **Types of ML**: Supervised, unsupervised, and reinforcement learning
- **Real-world Analogies**: Practical examples to understand ML concepts

### 2. Machine Learning Workflow
- **Data Collection and Understanding**: Loading and exploring datasets
- **Data Preprocessing**: Cleaning, handling missing values, and outlier detection
- **Feature Engineering**: Creating new features and transformations
- **Feature Selection**: Choosing the most relevant features
- **Model Training**: Training algorithms on prepared data
- **Model Evaluation**: Assessing model performance
- **Model Deployment**: Putting models into production

### 3. Data Preprocessing Techniques
- **Missing Value Handling**: Mean, median, mode imputation, and advanced methods
- **Outlier Detection**: Z-score, IQR, and advanced outlier detection methods
- **Feature Scaling**: Min-max scaling, standardization, and robust scaling
- **Categorical Encoding**: Label encoding, one-hot encoding, and target encoding

### 4. Feature Engineering
- **Polynomial Features**: Creating higher-order features
- **Interaction Features**: Combining multiple features
- **Time-based Features**: Creating temporal features
- **Domain-specific Features**: Business logic-based feature creation

### 5. Model Evaluation
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Regression Metrics**: MSE, MAE, R²
- **Cross-validation**: K-fold, stratified, and time series CV
- **ROC Curves and AUC**: Advanced evaluation techniques

### 6. Overfitting and Underfitting
- **Understanding the Concepts**: What they are and why they matter
- **Detection Methods**: How to identify overfitting and underfitting
- **Mitigation Strategies**: Regularization, cross-validation, and early stopping

### 7. Practical Example: Customer Churn Prediction
- **Complete ML Pipeline**: End-to-end implementation
- **Business Context**: Real-world application
- **Model Deployment**: Production-ready implementation

## Project Structure

```
chapter-03-ml-basics/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── aiprogramming/
│                   └── ch03/
│                       ├── MLBasicsDemo.java              # Main demonstration class
│                       ├── DataPoint.java                 # Data point representation
│                       ├── Dataset.java                   # Dataset management
│                       ├── MissingValueHandler.java       # Missing value handling
│                       ├── OutlierDetector.java           # Outlier detection
│                       ├── FeatureScaler.java             # Feature scaling
│                       ├── FeatureCreator.java            # Feature engineering
│                       ├── FeatureSelector.java           # Feature selection
│                       ├── ClassificationEvaluator.java   # Model evaluation
│                       ├── CrossValidator.java            # Cross-validation
│                       ├── CustomerChurnAnalysis.java     # Churn prediction example
│                       ├── ChurnDataPreprocessor.java     # Churn data preprocessing
│                       └── MLAlgorithm.java               # ML algorithm interfaces
├── pom.xml                                                # Maven configuration
└── README.md                                             # This file
```

## Key Classes and Their Purposes

### Core Data Structures
- **`DataPoint`**: Represents a single data point with features and target
- **`Dataset`**: Manages collections of data points with utility methods

### Data Preprocessing
- **`MissingValueHandler`**: Handles missing values using various strategies
- **`OutlierDetector`**: Detects and handles outliers using multiple methods
- **`FeatureScaler`**: Scales and normalizes features

### Feature Engineering
- **`FeatureCreator`**: Creates new features through various transformations
- **`FeatureSelector`**: Selects the most relevant features

### Model Evaluation
- **`ClassificationEvaluator`**: Evaluates classification models
- **`CrossValidator`**: Performs various types of cross-validation

### Practical Examples
- **`CustomerChurnAnalysis`**: Complete churn prediction pipeline
- **`ChurnDataPreprocessor`**: Specialized preprocessing for churn data
- **`MLBasicsDemo`**: Main demonstration class

## Getting Started

### Prerequisites
- Java 17 or higher
- Maven 3.6 or higher

### Building the Project
```bash
cd chapter-03-ml-basics
mvn clean compile
```

### Running the Examples
```bash
# Run the main demonstration
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch03.MLBasicsDemo"

# Or build and run the JAR
mvn clean package
java -jar target/chapter-03-ml-basics-1.0.0-jar-with-dependencies.jar
```

### Running Tests
```bash
mvn test
```

## Example Usage

### Basic Data Preprocessing
```java
// Load data
CustomerChurnAnalysis analysis = new CustomerChurnAnalysis();
Dataset rawData = analysis.loadCustomerData("customer_data.csv");

// Preprocess data
ChurnDataPreprocessor preprocessor = new ChurnDataPreprocessor();
Dataset processedData = preprocessor.preprocessData(rawData);

// Handle missing values
MissingValueHandler missingHandler = new MissingValueHandler();
Dataset cleanData = missingHandler.fillWithMean(processedData, "age");

// Scale features
FeatureScaler scaler = new FeatureScaler();
Dataset scaledData = scaler.standardize(cleanData, "income");
```

### Feature Engineering
```java
// Create polynomial features
FeatureCreator creator = new FeatureCreator();
Dataset withPolynomial = creator.createPolynomialFeatures(dataset, "age", 2);

// Create interaction features
Dataset withInteractions = creator.createInteractionFeatures(dataset, "age", "income");

// Feature selection
FeatureSelector selector = new FeatureSelector();
List<String> selectedFeatures = selector.selectByCorrelation(dataset, "target", 0.1);
```

### Model Evaluation
```java
// Split data
DatasetSplit split = dataset.split(0.8, 0.1, 0.1);

// Train model
SimpleClassifier model = new SimpleClassifier();
TrainedModel trainedModel = model.train(split.getTrainingSet());

// Evaluate
ClassificationEvaluator evaluator = new ClassificationEvaluator();
ClassificationMetrics metrics = evaluator.evaluate(trainedModel, split.getValidationSet());

System.out.println("Accuracy: " + metrics.getAccuracy());
System.out.println("F1-Score: " + metrics.getF1Score());
```

### Cross-Validation
```java
CrossValidator validator = new CrossValidator();
double cvScore = validator.crossValidate(dataset, model, 5);
System.out.println("Cross-validation score: " + cvScore);
```

## Customer Churn Prediction Example

The chapter includes a complete example of customer churn prediction that demonstrates:

1. **Data Loading**: Loading customer data from CSV files
2. **Data Exploration**: Understanding the dataset structure and characteristics
3. **Data Preprocessing**: Handling missing values, encoding categorical variables
4. **Feature Engineering**: Creating derived features and interactions
5. **Model Training**: Training multiple algorithms
6. **Model Evaluation**: Comparing different models
7. **Model Deployment**: Creating a prediction service

### Running the Churn Prediction Example
```java
// Load and analyze data
CustomerChurnAnalysis analysis = new CustomerChurnAnalysis();
Dataset rawData = analysis.loadCustomerData("customer_churn.csv");
analysis.exploreData(rawData);

// Preprocess data
ChurnDataPreprocessor preprocessor = new ChurnDataPreprocessor();
Dataset processedData = preprocessor.preprocessData(rawData);

// Train and evaluate models
// (See MLBasicsDemo.java for complete example)
```

## Key Concepts Covered

### Machine Learning Types
- **Supervised Learning**: Learning from labeled data
- **Unsupervised Learning**: Discovering patterns in unlabeled data
- **Reinforcement Learning**: Learning through interaction with environment

### Data Preprocessing Techniques
- **Missing Value Imputation**: Mean, median, mode, KNN, interpolation
- **Outlier Detection**: Z-score, IQR, modified Z-score, isolation forest
- **Feature Scaling**: Min-max, standardization, robust scaling
- **Categorical Encoding**: Label encoding, one-hot encoding, target encoding

### Feature Engineering Methods
- **Polynomial Features**: Creating higher-order terms
- **Interaction Features**: Combining multiple features
- **Time-based Features**: Temporal feature creation
- **Domain Features**: Business logic-based features

### Model Evaluation Metrics
- **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Regression**: MSE, MAE, R², RMSE
- **Cross-validation**: K-fold, stratified, leave-one-out, time series

### Overfitting/Underfitting
- **Detection**: Learning curves, validation metrics
- **Mitigation**: Regularization, early stopping, feature selection

## Exercises

The chapter includes several exercises to reinforce learning:

1. **Data Preprocessing Pipeline**: Implement comprehensive preprocessing
2. **Model Evaluation Framework**: Build evaluation utilities
3. **Customer Segmentation**: Unsupervised learning example
4. **Feature Engineering Challenge**: Advanced feature creation
5. **End-to-End ML Project**: Complete pipeline implementation

## Next Steps

After completing this chapter, you'll be ready to explore:

- **Chapter 4**: Supervised Learning - Classification algorithms
- **Chapter 5**: Supervised Learning - Regression algorithms
- **Chapter 6**: Unsupervised Learning - Clustering and dimensionality reduction

## Contributing

Feel free to contribute improvements, bug fixes, or additional examples. Please ensure all code follows the existing style and includes appropriate tests.

## License

This project is part of the "AI Programming with Java" book and follows the same licensing terms.

# Chapter 3: Introduction to Machine Learning

Welcome to Chapter 3 of **AI Programming with Java**! This chapter introduces the fundamental concepts of machine learning, the core technology that powers most modern AI applications. We'll explore what machine learning is, how it works, and how to implement basic ML concepts using Java.

## Learning Objectives

- Understand the core concepts and principles of machine learning
- Learn about different types of ML problems and their applications
- Master the ML workflow from data to prediction
- Implement data preprocessing and feature engineering techniques
- Understand model evaluation metrics and validation strategies
- Learn to identify and handle overfitting and underfitting
- Build practical ML applications using Java

## What is Machine Learning?

Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario. Instead of following rigid rules, ML systems identify patterns in data and use these patterns to make predictions or decisions.

### The Essence of Machine Learning

**Traditional Programming vs. Machine Learning:**

**Traditional Programming:**
```
Input Data + Rules → Output
```

**Machine Learning:**
```
Input Data + Output → Rules (Model)
```

Then:
```
New Input Data + Rules (Model) → New Output
```

### Real-World Analogies

**Learning to Ride a Bicycle (Supervised Learning):**
- You observe others riding bikes (training data)
- You try to ride and get feedback (labels/outcomes)
- You adjust your technique based on success/failure
- Eventually, you can ride without thinking about it

**Organizing a Library (Unsupervised Learning):**
- You have books with no predefined categories
- You group similar books together based on content
- You create categories that make sense
- The organization helps others find books

**Training a Dog (Reinforcement Learning):**
- Dog performs actions in an environment
- You provide rewards for good behavior
- Dog learns to maximize rewards
- Dog adapts behavior based on consequences

## Types of Machine Learning

Machine learning can be categorized into three main types, each with distinct characteristics and applications.

### 1. Supervised Learning

Supervised learning involves training a model on labeled data, where each example has a known outcome or target value.

**Key Characteristics:**
- Uses labeled training data
- Learns to map inputs to outputs
- Can make predictions on new, unseen data
- Includes both classification and regression problems

**Common Applications:**
- Email spam detection (classification)
- House price prediction (regression)
- Medical diagnosis (classification)
- Stock price forecasting (regression)

**Example: Email Classification**
```java
// Training data with labels
Email email1 = new Email("Buy now! Limited time offer!", "spam");
Email email2 = new Email("Meeting tomorrow at 2 PM", "ham");
Email email3 = new Email("Your order has been shipped", "ham");
// ... more examples

// Model learns patterns from these examples
// Then can classify new emails as spam or ham
```

### 2. Unsupervised Learning

Unsupervised learning works with unlabeled data to discover hidden patterns or structures.

**Key Characteristics:**
- Uses unlabeled data
- Discovers patterns and structures
- No predefined outcomes
- Includes clustering and dimensionality reduction

**Common Applications:**
- Customer segmentation (clustering)
- Image compression (dimensionality reduction)
- Market basket analysis (association rules)
- Anomaly detection

**Example: Customer Segmentation**
```java
// Customer data without predefined segments
Customer customer1 = new Customer(25, 50000, "urban", 5);
Customer customer2 = new Customer(45, 120000, "suburban", 12);
Customer customer3 = new Customer(32, 75000, "urban", 8);
// ... more customers

// Model discovers natural groupings
// Result: Young urban professionals, affluent suburban families, etc.
```

### 3. Reinforcement Learning

Reinforcement learning involves an agent learning to make decisions by interacting with an environment and receiving rewards or penalties.

**Key Characteristics:**
- Agent interacts with environment
- Learns through trial and error
- Maximizes cumulative rewards
- Balances exploration and exploitation

**Common Applications:**
- Game AI (chess, Go, video games)
- Autonomous vehicles
- Robot navigation
- Trading algorithms

**Example: Game AI**
```java
// Agent in a game environment
GameState state = new GameState();
Action action = agent.chooseAction(state);
GameState newState = environment.step(action);
double reward = environment.getReward();
agent.learn(state, action, reward, newState);
```

## The Machine Learning Workflow

The ML workflow is a systematic process that transforms raw data into actionable insights through a series of well-defined steps.

### 1. Data Collection and Understanding

**Data Sources:**
- Databases and data warehouses
- APIs and web scraping
- Sensors and IoT devices
- User interactions and logs
- External datasets and repositories

**Data Understanding:**
- Data types and formats
- Data quality assessment
- Missing values and outliers
- Data distribution analysis
- Domain knowledge integration

**Example: Data Collection Framework**
```java
public class DataCollector {
    private final List<DataSource> dataSources;
    
    public DataCollector(List<DataSource> dataSources) {
        this.dataSources = dataSources;
    }
    
    public Dataset collectData() {
        List<DataPoint> allData = new ArrayList<>();
        
        for (DataSource source : dataSources) {
            List<DataPoint> sourceData = source.fetchData();
            allData.addAll(sourceData);
        }
        
        return new Dataset(allData);
    }
}

public interface DataSource {
    List<DataPoint> fetchData();
}

// Example implementations
public class DatabaseSource implements DataSource {
    @Override
    public List<DataPoint> fetchData() {
        // Fetch data from database
        return database.query("SELECT * FROM customer_data");
    }
}

public class APISource implements DataSource {
    @Override
    public List<DataPoint> fetchData() {
        // Fetch data from REST API
        return apiClient.get("/api/customers");
    }
}
```

### 2. Data Preprocessing and Cleaning

Data preprocessing is crucial for ensuring data quality and preparing it for machine learning algorithms.

**Common Preprocessing Steps:**

**Handling Missing Values:**
```java
public class MissingValueHandler {
    
    // Remove rows with missing values
    public Dataset removeMissingValues(Dataset dataset) {
        return dataset.filter(point -> !hasMissingValues(point));
    }
    
    // Fill missing values with mean
    public Dataset fillWithMean(Dataset dataset, String featureName) {
        double mean = calculateMean(dataset, featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, mean);
            }
            return point;
        });
    }
    
    // Fill missing values with median
    public Dataset fillWithMedian(Dataset dataset, String featureName) {
        double median = calculateMedian(dataset, featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, median);
            }
            return point;
        });
    }
    
    // Fill missing values with mode
    public Dataset fillWithMode(Dataset dataset, String featureName) {
        Object mode = calculateMode(dataset, featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, mode);
            }
            return point;
        });
    }
}
```

**Outlier Detection and Handling:**
```java
public class OutlierDetector {
    
    // Z-score method for outlier detection
    public List<DataPoint> detectOutliersZScore(Dataset dataset, String featureName, double threshold) {
        double mean = calculateMean(dataset, featureName);
        double std = calculateStandardDeviation(dataset, featureName);
        
        return dataset.filter(point -> {
            double value = point.getFeature(featureName);
            double zScore = Math.abs((value - mean) / std);
            return zScore > threshold;
        }).collect(Collectors.toList());
    }
    
    // IQR method for outlier detection
    public List<DataPoint> detectOutliersIQR(Dataset dataset, String featureName, double multiplier) {
        double q1 = calculatePercentile(dataset, featureName, 25);
        double q3 = calculatePercentile(dataset, featureName, 75);
        double iqr = q3 - q1;
        
        double lowerBound = q1 - multiplier * iqr;
        double upperBound = q3 + multiplier * iqr;
        
        return dataset.filter(point -> {
            double value = point.getFeature(featureName);
            return value < lowerBound || value > upperBound;
        }).collect(Collectors.toList());
    }
}
```

**Data Type Conversion:**
```java
public class DataTypeConverter {
    
    // Convert categorical to numerical
    public Dataset encodeCategorical(Dataset dataset, String featureName) {
        Map<String, Integer> encoding = createEncoding(dataset, featureName);
        
        return dataset.map(point -> {
            String categoricalValue = point.getFeature(featureName);
            Integer numericalValue = encoding.get(categoricalValue);
            point.setFeature(featureName, numericalValue);
            return point;
        });
    }
    
    // Convert numerical to categorical (binning)
    public Dataset binNumerical(Dataset dataset, String featureName, int numBins) {
        double min = dataset.getMin(featureName);
        double max = dataset.getMax(featureName);
        double binSize = (max - min) / numBins;
        
        return dataset.map(point -> {
            double value = point.getFeature(featureName);
            int bin = (int) ((value - min) / binSize);
            if (bin >= numBins) bin = numBins - 1;
            point.setFeature(featureName, "bin_" + bin);
            return point;
        });
    }
}
```

### 3. Feature Engineering

Feature engineering is the process of creating new features or modifying existing ones to improve model performance.

**Feature Scaling and Normalization:**
```java
public class FeatureScaler {
    
    // Min-Max scaling to [0, 1] range
    public Dataset minMaxScale(Dataset dataset, String featureName) {
        double min = dataset.getMin(featureName);
        double max = dataset.getMax(featureName);
        
        return dataset.map(point -> {
            double value = point.getFeature(featureName);
            double scaledValue = (value - min) / (max - min);
            point.setFeature(featureName, scaledValue);
            return point;
        });
    }
    
    // Standardization (Z-score normalization)
    public Dataset standardize(Dataset dataset, String featureName) {
        double mean = calculateMean(dataset, featureName);
        double std = calculateStandardDeviation(dataset, featureName);
        
        return dataset.map(point -> {
            double value = point.getFeature(featureName);
            double standardizedValue = (value - mean) / std;
            point.setFeature(featureName, standardizedValue);
            return point;
        });
    }
    
    // Robust scaling using median and IQR
    public Dataset robustScale(Dataset dataset, String featureName) {
        double median = calculateMedian(dataset, featureName);
        double q1 = calculatePercentile(dataset, featureName, 25);
        double q3 = calculatePercentile(dataset, featureName, 75);
        double iqr = q3 - q1;
        
        return dataset.map(point -> {
            double value = point.getFeature(featureName);
            double scaledValue = (value - median) / iqr;
            point.setFeature(featureName, scaledValue);
            return point;
        });
    }
}
```

**Feature Creation:**
```java
public class FeatureCreator {
    
    // Create polynomial features
    public Dataset createPolynomialFeatures(Dataset dataset, String featureName, int degree) {
        return dataset.map(point -> {
            double value = point.getFeature(featureName);
            for (int i = 2; i <= degree; i++) {
                String newFeatureName = featureName + "_pow_" + i;
                point.setFeature(newFeatureName, Math.pow(value, i));
            }
            return point;
        });
    }
    
    // Create interaction features
    public Dataset createInteractionFeatures(Dataset dataset, String feature1, String feature2) {
        return dataset.map(point -> {
            double value1 = point.getFeature(feature1);
            double value2 = point.getFeature(feature2);
            String interactionName = feature1 + "_x_" + feature2;
            point.setFeature(interactionName, value1 * value2);
            return point;
        });
    }
    
    // Create time-based features
    public Dataset createTimeFeatures(Dataset dataset, String dateFeature) {
        return dataset.map(point -> {
            LocalDate date = point.getFeature(dateFeature);
            
            point.setFeature("day_of_week", date.getDayOfWeek().getValue());
            point.setFeature("month", date.getMonthValue());
            point.setFeature("quarter", (date.getMonthValue() - 1) / 3 + 1);
            point.setFeature("is_weekend", 
                date.getDayOfWeek() == DayOfWeek.SATURDAY || 
                date.getDayOfWeek() == DayOfWeek.SUNDAY ? 1 : 0);
            
            return point;
        });
    }
}
```

### 4. Feature Selection

Feature selection helps reduce dimensionality and improve model performance by selecting the most relevant features.

**Statistical Methods:**
```java
public class FeatureSelector {
    
    // Correlation-based feature selection
    public List<String> selectByCorrelation(Dataset dataset, String targetFeature, double threshold) {
        List<String> selectedFeatures = new ArrayList<>();
        
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double correlation = calculateCorrelation(dataset, feature, targetFeature);
                if (Math.abs(correlation) > threshold) {
                    selectedFeatures.add(feature);
                }
            }
        }
        
        return selectedFeatures;
    }
    
    // Variance-based feature selection
    public List<String> selectByVariance(Dataset dataset, double threshold) {
        return dataset.getFeatureNames().stream()
            .filter(feature -> calculateVariance(dataset, feature) > threshold)
            .collect(Collectors.toList());
    }
    
    // Mutual information-based selection
    public List<String> selectByMutualInformation(Dataset dataset, String targetFeature, int topK) {
        Map<String, Double> miScores = new HashMap<>();
        
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double mi = calculateMutualInformation(dataset, feature, targetFeature);
                miScores.put(feature, mi);
            }
        }
        
        return miScores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
}
```

### 5. Model Training

Model training involves selecting an appropriate algorithm and optimizing its parameters.

**Training Pipeline:**
```java
public class ModelTrainer {
    
    public TrainedModel trainModel(Dataset trainingData, MLAlgorithm algorithm, 
                                 Map<String, Object> hyperparameters) {
        
        // Initialize the algorithm with hyperparameters
        algorithm.setHyperparameters(hyperparameters);
        
        // Train the model
        long startTime = System.currentTimeMillis();
        TrainedModel model = algorithm.train(trainingData);
        long endTime = System.currentTimeMillis();
        
        System.out.println("Training completed in " + (endTime - startTime) + " ms");
        
        return model;
    }
    
    // Cross-validation training
    public CrossValidationResults crossValidate(Dataset dataset, MLAlgorithm algorithm, 
                                              Map<String, Object> hyperparameters, int folds) {
        
        List<Double> foldScores = new ArrayList<>();
        List<Dataset> folds = createFolds(dataset, folds);
        
        for (int i = 0; i < folds; i++) {
            // Create training and validation sets
            Dataset validationSet = folds.get(i);
            Dataset trainingSet = combineFolds(folds, i);
            
            // Train model
            TrainedModel model = trainModel(trainingSet, algorithm, hyperparameters);
            
            // Evaluate on validation set
            double score = model.evaluate(validationSet);
            foldScores.add(score);
        }
        
        return new CrossValidationResults(foldScores);
    }
}
```

### 6. Model Evaluation

Model evaluation assesses how well the trained model performs on unseen data.

**Classification Metrics:**
```java
public class ClassificationEvaluator {
    
    public ClassificationMetrics evaluate(TrainedModel model, Dataset testData) {
        List<Prediction> predictions = model.predict(testData);
        List<Object> actualLabels = testData.getLabels();
        
        // Calculate confusion matrix
        Map<String, Map<String, Integer>> confusionMatrix = calculateConfusionMatrix(
            predictions, actualLabels);
        
        // Calculate metrics
        double accuracy = calculateAccuracy(confusionMatrix);
        double precision = calculatePrecision(confusionMatrix);
        double recall = calculateRecall(confusionMatrix);
        double f1Score = calculateF1Score(precision, recall);
        
        return new ClassificationMetrics(accuracy, precision, recall, f1Score, confusionMatrix);
    }
    
    private double calculateAccuracy(Map<String, Map<String, Integer>> confusionMatrix) {
        int correct = 0;
        int total = 0;
        
        for (String predicted : confusionMatrix.keySet()) {
            for (String actual : confusionMatrix.get(predicted).keySet()) {
                int count = confusionMatrix.get(predicted).get(actual);
                if (predicted.equals(actual)) {
                    correct += count;
                }
                total += count;
            }
        }
        
        return (double) correct / total;
    }
    
    private double calculatePrecision(Map<String, Map<String, Integer>> confusionMatrix) {
        // Calculate precision for each class and return macro-average
        double totalPrecision = 0.0;
        int numClasses = confusionMatrix.size();
        
        for (String predicted : confusionMatrix.keySet()) {
            int truePositives = confusionMatrix.get(predicted).getOrDefault(predicted, 0);
            int falsePositives = 0;
            
            for (String actual : confusionMatrix.keySet()) {
                if (!actual.equals(predicted)) {
                    falsePositives += confusionMatrix.get(predicted).getOrDefault(actual, 0);
                }
            }
            
            double precision = (truePositives + falsePositives) > 0 ? 
                (double) truePositives / (truePositives + falsePositives) : 0.0;
            totalPrecision += precision;
        }
        
        return totalPrecision / numClasses;
    }
}
```

**Regression Metrics:**
```java
public class RegressionEvaluator {
    
    public RegressionMetrics evaluate(TrainedModel model, Dataset testData) {
        List<Prediction> predictions = model.predict(testData);
        List<Double> actualValues = testData.getTargetValues();
        
        double mse = calculateMSE(predictions, actualValues);
        double mae = calculateMAE(predictions, actualValues);
        double rmse = Math.sqrt(mse);
        double r2 = calculateR2(predictions, actualValues);
        
        return new RegressionMetrics(mse, mae, rmse, r2);
    }
    
    private double calculateMSE(List<Prediction> predictions, List<Double> actualValues) {
        double sumSquaredError = 0.0;
        
        for (int i = 0; i < predictions.size(); i++) {
            double predicted = predictions.get(i).getValue();
            double actual = actualValues.get(i);
            double error = predicted - actual;
            sumSquaredError += error * error;
        }
        
        return sumSquaredError / predictions.size();
    }
    
    private double calculateMAE(List<Prediction> predictions, List<Double> actualValues) {
        double sumAbsoluteError = 0.0;
        
        for (int i = 0; i < predictions.size(); i++) {
            double predicted = predictions.get(i).getValue();
            double actual = actualValues.get(i);
            sumAbsoluteError += Math.abs(predicted - actual);
        }
        
        return sumAbsoluteError / predictions.size();
    }
    
    private double calculateR2(List<Prediction> predictions, List<Double> actualValues) {
        double mean = actualValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double ssRes = 0.0; // Sum of squared residuals
        double ssTot = 0.0; // Total sum of squares
        
        for (int i = 0; i < predictions.size(); i++) {
            double predicted = predictions.get(i).getValue();
            double actual = actualValues.get(i);
            
            ssRes += Math.pow(predicted - actual, 2);
            ssTot += Math.pow(actual - mean, 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
}
```

### 7. Model Deployment and Monitoring

Once a model is trained and evaluated, it needs to be deployed and monitored in production.

**Model Serialization:**
```java
public class ModelSerializer {
    
    public void saveModel(TrainedModel model, String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(model);
            System.out.println("Model saved to: " + filePath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save model", e);
        }
    }
    
    public TrainedModel loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            TrainedModel model = (TrainedModel) ois.readObject();
            System.out.println("Model loaded from: " + filePath);
            return model;
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }
}
```

**Model Serving:**
```java
public class ModelServer {
    private final TrainedModel model;
    private final FeatureProcessor featureProcessor;
    
    public ModelServer(TrainedModel model, FeatureProcessor featureProcessor) {
        this.model = model;
        this.featureProcessor = featureProcessor;
    }
    
    public PredictionResponse predict(PredictionRequest request) {
        try {
            // Preprocess input features
            DataPoint processedData = featureProcessor.process(request.getFeatures());
            
            // Make prediction
            Prediction prediction = model.predict(processedData);
            
            return new PredictionResponse(prediction, "SUCCESS");
        } catch (Exception e) {
            return new PredictionResponse(null, "ERROR: " + e.getMessage());
        }
    }
    
    public ModelHealth checkHealth() {
        // Perform health checks
        boolean modelLoaded = model != null;
        boolean canPredict = modelLoaded && model.isReady();
        
        return new ModelHealth(modelLoaded, canPredict, System.currentTimeMillis());
    }
}
```

## Overfitting and Underfitting

Two critical challenges in machine learning are overfitting and underfitting, which can significantly impact model performance.

### Understanding Overfitting

**What is Overfitting?**
Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor generalization on new data.

**Signs of Overfitting:**
- High training accuracy, low validation accuracy
- Model performs well on training data but poorly on test data
- Complex model with many parameters
- Model memorizes training examples

**Example: Overfitting in Polynomial Regression**
```java
public class OverfittingExample {
    
    public void demonstrateOverfitting() {
        // Generate synthetic data with noise
        List<DataPoint> trainingData = generateSyntheticData(20, 0.1);
        List<DataPoint> testData = generateSyntheticData(100, 0.1);
        
        // Train models with different polynomial degrees
        for (int degree = 1; degree <= 10; degree++) {
            PolynomialRegression model = new PolynomialRegression(degree);
            model.train(trainingData);
            
            double trainingError = model.evaluate(trainingData);
            double testError = model.evaluate(testData);
            
            System.out.printf("Degree %d: Training Error=%.4f, Test Error=%.4f%n", 
                degree, trainingError, testError);
        }
    }
    
    private List<DataPoint> generateSyntheticData(int numPoints, double noiseLevel) {
        List<DataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numPoints; i++) {
            double x = random.nextDouble() * 10;
            double y = 2 * x + 1 + random.nextGaussian() * noiseLevel; // Linear relationship + noise
            data.add(new DataPoint(x, y));
        }
        
        return data;
    }
}
```

### Understanding Underfitting

**What is Underfitting?**
Underfitting occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test data.

**Signs of Underfitting:**
- Low training accuracy and low validation accuracy
- Model is too simple for the complexity of the data
- High bias, low variance
- Model fails to learn meaningful patterns

**Example: Underfitting in Classification**
```java
public class UnderfittingExample {
    
    public void demonstrateUnderfitting() {
        // Generate complex non-linear data
        List<DataPoint> trainingData = generateComplexData(100);
        List<DataPoint> testData = generateComplexData(50);
        
        // Try linear model on non-linear data
        LinearClassifier linearModel = new LinearClassifier();
        linearModel.train(trainingData);
        
        double trainingAccuracy = linearModel.evaluate(trainingData);
        double testAccuracy = linearModel.evaluate(testData);
        
        System.out.printf("Linear Model: Training Accuracy=%.4f, Test Accuracy=%.4f%n", 
            trainingAccuracy, testAccuracy);
        
        // Try non-linear model
        NonLinearClassifier nonLinearModel = new NonLinearClassifier();
        nonLinearModel.train(trainingData);
        
        trainingAccuracy = nonLinearModel.evaluate(trainingData);
        testAccuracy = nonLinearModel.evaluate(testData);
        
        System.out.printf("Non-Linear Model: Training Accuracy=%.4f, Test Accuracy=%.4f%n", 
            trainingAccuracy, testAccuracy);
    }
    
    private List<DataPoint> generateComplexData(int numPoints) {
        List<DataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numPoints; i++) {
            double x1 = random.nextDouble() * 4 - 2;
            double x2 = random.nextDouble() * 4 - 2;
            
            // XOR-like pattern
            int label = (x1 > 0 && x2 > 0) || (x1 < 0 && x2 < 0) ? 1 : 0;
            
            data.add(new DataPoint(new double[]{x1, x2}, label));
        }
        
        return data;
    }
}
```

### Strategies to Address Overfitting and Underfitting

**For Overfitting:**
```java
public class OverfittingSolutions {
    
    // 1. Regularization
    public class RegularizedModel {
        private final double lambda; // Regularization parameter
        
        public RegularizedModel(double lambda) {
            this.lambda = lambda;
        }
        
        public double calculateLoss(double[] predictions, double[] actual, double[] weights) {
            double dataLoss = calculateDataLoss(predictions, actual);
            double regularizationLoss = lambda * calculateRegularizationLoss(weights);
            return dataLoss + regularizationLoss;
        }
        
        private double calculateRegularizationLoss(double[] weights) {
            // L2 regularization (Ridge)
            return Arrays.stream(weights).map(w -> w * w).sum();
        }
    }
    
    // 2. Cross-validation
    public class CrossValidator {
        public double crossValidate(Dataset dataset, MLAlgorithm algorithm, int folds) {
            List<Double> foldScores = new ArrayList<>();
            List<Dataset> folds = createFolds(dataset, folds);
            
            for (int i = 0; i < folds; i++) {
                Dataset validationSet = folds.get(i);
                Dataset trainingSet = combineFolds(folds, i);
                
                TrainedModel model = algorithm.train(trainingSet);
                double score = model.evaluate(validationSet);
                foldScores.add(score);
            }
            
            return foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        }
    }
    
    // 3. Early stopping
    public class EarlyStoppingTrainer {
        public TrainedModel trainWithEarlyStopping(Dataset trainingData, Dataset validationData, 
                                                 MLAlgorithm algorithm, int maxEpochs, int patience) {
            
            TrainedModel bestModel = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            int epochsWithoutImprovement = 0;
            
            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                TrainedModel currentModel = algorithm.trainForEpoch(trainingData, epoch);
                double currentScore = currentModel.evaluate(validationData);
                
                if (currentScore > bestScore) {
                    bestScore = currentScore;
                    bestModel = currentModel;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                }
                
                if (epochsWithoutImprovement >= patience) {
                    System.out.println("Early stopping at epoch " + epoch);
                    break;
                }
            }
            
            return bestModel;
        }
    }
}
```

**For Underfitting:**
```java
public class UnderfittingSolutions {
    
    // 1. Increase model complexity
    public class ComplexModel {
        private final int numLayers;
        private final int neuronsPerLayer;
        
        public ComplexModel(int numLayers, int neuronsPerLayer) {
            this.numLayers = numLayers;
            this.neuronsPerLayer = neuronsPerLayer;
        }
        
        public TrainedModel train(Dataset dataset) {
            // Implement more complex model architecture
            return new NeuralNetwork(numLayers, neuronsPerLayer).train(dataset);
        }
    }
    
    // 2. Feature engineering
    public class FeatureEngineer {
        public Dataset engineerFeatures(Dataset dataset) {
            Dataset enhanced = dataset.copy();
            
            // Add polynomial features
            for (String feature : dataset.getNumericalFeatures()) {
                enhanced = addPolynomialFeatures(enhanced, feature, 3);
            }
            
            // Add interaction features
            List<String> numericalFeatures = dataset.getNumericalFeatures();
            for (int i = 0; i < numericalFeatures.size(); i++) {
                for (int j = i + 1; j < numericalFeatures.size(); j++) {
                    enhanced = addInteractionFeature(enhanced, 
                        numericalFeatures.get(i), numericalFeatures.get(j));
                }
            }
            
            return enhanced;
        }
    }
    
    // 3. Reduce regularization
    public class ReducedRegularization {
        public TrainedModel trainWithReducedRegularization(Dataset dataset, MLAlgorithm algorithm) {
            // Use smaller regularization parameter
            Map<String, Object> hyperparameters = new HashMap<>();
            hyperparameters.put("lambda", 0.001); // Smaller regularization
            hyperparameters.put("learningRate", 0.1); // Larger learning rate
            
            return algorithm.train(dataset, hyperparameters);
        }
    }
}
```

## Practical Example: Customer Churn Prediction

Let's implement a complete machine learning pipeline for predicting customer churn using the concepts we've learned.

### Problem Definition

**Business Context:**
A telecommunications company wants to predict which customers are likely to cancel their service (churn) so they can take proactive measures to retain them.

**Data Description:**
- Customer demographics (age, gender, location)
- Service usage patterns (minutes, data usage, calls)
- Billing information (monthly charges, payment method)
- Customer service interactions (complaints, support calls)

### Data Loading and Exploration
```java
public class CustomerChurnAnalysis {
    
    public Dataset loadCustomerData(String filePath) {
        List<DataPoint> dataPoints = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line = reader.readLine(); // Skip header
            while ((line = reader.readLine()) != null) {
                DataPoint point = parseCustomerData(line);
                dataPoints.add(point);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load customer data", e);
        }
        
        return new Dataset(dataPoints);
    }
    
    private DataPoint parseCustomerData(String line) {
        String[] fields = line.split(",");
        
        // Parse features
        int customerId = Integer.parseInt(fields[0]);
        String gender = fields[1];
        int age = Integer.parseInt(fields[2]);
        String location = fields[3];
        int tenure = Integer.parseInt(fields[4]);
        double monthlyCharges = Double.parseDouble(fields[5]);
        double totalCharges = Double.parseDouble(fields[6]);
        String contractType = fields[7];
        String paymentMethod = fields[8];
        int supportCalls = Integer.parseInt(fields[9]);
        boolean hasChurned = fields[10].equals("Yes");
        
        // Create feature vector
        Map<String, Object> features = new HashMap<>();
        features.put("customerId", customerId);
        features.put("gender", gender);
        features.put("age", age);
        features.put("location", location);
        features.put("tenure", tenure);
        features.put("monthlyCharges", monthlyCharges);
        features.put("totalCharges", totalCharges);
        features.put("contractType", contractType);
        features.put("paymentMethod", paymentMethod);
        features.put("supportCalls", supportCalls);
        
        return new DataPoint(features, hasChurned);
    }
    
    public void exploreData(Dataset dataset) {
        System.out.println("=== Customer Churn Dataset Exploration ===");
        System.out.println("Total customers: " + dataset.size());
        System.out.println("Features: " + dataset.getFeatureNames());
        
        // Churn distribution
        long churnedCount = dataset.getDataPoints().stream()
            .filter(point -> (Boolean) point.getTarget())
            .count();
        double churnRate = (double) churnedCount / dataset.size();
        System.out.printf("Churn rate: %.2f%%%n", churnRate * 100);
        
        // Feature statistics
        for (String feature : dataset.getNumericalFeatures()) {
            double mean = calculateMean(dataset, feature);
            double std = calculateStandardDeviation(dataset, feature);
            System.out.printf("%s: mean=%.2f, std=%.2f%n", feature, mean, std);
        }
    }
}
```

### Data Preprocessing
```java
public class ChurnDataPreprocessor {
    
    public Dataset preprocessData(Dataset dataset) {
        Dataset processed = dataset.copy();
        
        // Handle missing values
        processed = handleMissingValues(processed);
        
        // Encode categorical variables
        processed = encodeCategoricalVariables(processed);
        
        // Scale numerical features
        processed = scaleNumericalFeatures(processed);
        
        // Create new features
        processed = createDerivedFeatures(processed);
        
        return processed;
    }
    
    private Dataset handleMissingValues(Dataset dataset) {
        // Fill missing total charges with monthly charges * tenure
        return dataset.map(point -> {
            if (point.getFeature("totalCharges") == null) {
                double monthlyCharges = point.getFeature("monthlyCharges");
                int tenure = point.getFeature("tenure");
                point.setFeature("totalCharges", monthlyCharges * tenure);
            }
            return point;
        });
    }
    
    private Dataset encodeCategoricalVariables(Dataset dataset) {
        // Encode gender
        dataset = encodeBinary(dataset, "gender", "Male", "Female");
        
        // Encode contract type
        dataset = encodeCategorical(dataset, "contractType", 
            Arrays.asList("Month-to-month", "One year", "Two year"));
        
        // Encode payment method
        dataset = encodeCategorical(dataset, "paymentMethod",
            Arrays.asList("Electronic check", "Mailed check", "Bank transfer", "Credit card"));
        
        return dataset;
    }
    
    private Dataset scaleNumericalFeatures(Dataset dataset) {
        String[] numericalFeatures = {"age", "tenure", "monthlyCharges", "totalCharges", "supportCalls"};
        
        for (String feature : numericalFeatures) {
            dataset = standardize(dataset, feature);
        }
        
        return dataset;
    }
    
    private Dataset createDerivedFeatures(Dataset dataset) {
        return dataset.map(point -> {
            // Average monthly charges
            double monthlyCharges = point.getFeature("monthlyCharges");
            int tenure = point.getFeature("tenure");
            double avgMonthlyCharges = tenure > 0 ? monthlyCharges / tenure : monthlyCharges;
            point.setFeature("avgMonthlyCharges", avgMonthlyCharges);
            
            // Tenure to age ratio
            int age = point.getFeature("age");
            double tenureToAgeRatio = age > 0 ? (double) tenure / age : 0.0;
            point.setFeature("tenureToAgeRatio", tenureToAgeRatio);
            
            // High support calls flag
            int supportCalls = point.getFeature("supportCalls");
            point.setFeature("highSupportCalls", supportCalls > 3 ? 1 : 0);
            
            return point;
        });
    }
}
```

### Model Training and Evaluation
```java
public class ChurnPredictionModel {
    
    public void trainAndEvaluateChurnModel() {
        // Load and preprocess data
        CustomerChurnAnalysis analysis = new CustomerChurnAnalysis();
        Dataset rawData = analysis.loadCustomerData("customer_churn.csv");
        
        ChurnDataPreprocessor preprocessor = new ChurnDataPreprocessor();
        Dataset processedData = preprocessor.preprocessData(rawData);
        
        // Split data
        DatasetSplit split = processedData.split(0.8, 0.1, 0.1); // train, validation, test
        
        // Train multiple models
        List<MLAlgorithm> algorithms = Arrays.asList(
            new LogisticRegression(),
            new RandomForest(),
            new SupportVectorMachine(),
            new GradientBoosting()
        );
        
        Map<String, TrainedModel> models = new HashMap<>();
        Map<String, ClassificationMetrics> results = new HashMap<>();
        
        for (MLAlgorithm algorithm : algorithms) {
            String modelName = algorithm.getClass().getSimpleName();
            System.out.println("Training " + modelName + "...");
            
            // Train model
            TrainedModel model = algorithm.train(split.getTrainingSet());
            models.put(modelName, model);
            
            // Evaluate on validation set
            ClassificationMetrics metrics = new ClassificationEvaluator()
                .evaluate(model, split.getValidationSet());
            results.put(modelName, metrics);
            
            System.out.printf("%s - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f%n",
                modelName, metrics.getAccuracy(), metrics.getPrecision(), 
                metrics.getRecall(), metrics.getF1Score());
        }
        
        // Select best model
        String bestModelName = selectBestModel(results);
        TrainedModel bestModel = models.get(bestModelName);
        
        // Final evaluation on test set
        ClassificationMetrics finalMetrics = new ClassificationEvaluator()
            .evaluate(bestModel, split.getTestSet());
        
        System.out.println("\n=== Final Results ===");
        System.out.printf("Best Model: %s%n", bestModelName);
        System.out.printf("Test Accuracy: %.4f%n", finalMetrics.getAccuracy());
        System.out.printf("Test Precision: %.4f%n", finalMetrics.getPrecision());
        System.out.printf("Test Recall: %.4f%n", finalMetrics.getRecall());
        System.out.printf("Test F1-Score: %.4f%n", finalMetrics.getF1Score());
        
        // Save model
        new ModelSerializer().saveModel(bestModel, "churn_prediction_model.ser");
    }
    
    private String selectBestModel(Map<String, ClassificationMetrics> results) {
        return results.entrySet().stream()
            .max(Comparator.comparing(entry -> entry.getValue().getF1Score()))
            .map(Map.Entry::getKey)
            .orElse("Unknown");
    }
}
```

### Model Deployment
```java
public class ChurnPredictionService {
    private final TrainedModel model;
    private final FeatureProcessor featureProcessor;
    
    public ChurnPredictionService(String modelPath) {
        this.model = new ModelSerializer().loadModel(modelPath);
        this.featureProcessor = new ChurnDataPreprocessor();
    }
    
    public ChurnPrediction predictChurn(CustomerData customerData) {
        try {
            // Preprocess customer data
            DataPoint processedData = featureProcessor.process(customerData);
            
            // Make prediction
            Prediction prediction = model.predict(processedData);
            
            // Calculate churn probability
            double churnProbability = prediction.getProbability();
            boolean willChurn = churnProbability > 0.5;
            
            // Generate recommendations
            List<String> recommendations = generateRecommendations(customerData, churnProbability);
            
            return new ChurnPrediction(willChurn, churnProbability, recommendations);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to predict churn", e);
        }
    }
    
    private List<String> generateRecommendations(CustomerData customerData, double churnProbability) {
        List<String> recommendations = new ArrayList<>();
        
        if (churnProbability > 0.7) {
            recommendations.add("High churn risk - immediate intervention required");
            recommendations.add("Offer loyalty discount or retention package");
            recommendations.add("Assign dedicated customer success manager");
        } else if (churnProbability > 0.5) {
            recommendations.add("Moderate churn risk - proactive outreach recommended");
            recommendations.add("Send personalized retention offers");
            recommendations.add("Schedule follow-up call within 48 hours");
        } else {
            recommendations.add("Low churn risk - maintain current service quality");
            recommendations.add("Continue regular engagement activities");
        }
        
        // Add specific recommendations based on customer characteristics
        if (customerData.getSupportCalls() > 3) {
            recommendations.add("Address customer service issues promptly");
        }
        
        if (customerData.getMonthlyCharges() > 100) {
            recommendations.add("Consider offering premium service benefits");
        }
        
        return recommendations;
    }
}
```

## Summary

In this chapter, we've explored the fundamental concepts of machine learning and how to implement them using Java. Here's what we've covered:

### Key Concepts
- **Machine Learning Types**: Supervised, unsupervised, and reinforcement learning
- **ML Workflow**: From data collection to model deployment
- **Data Preprocessing**: Cleaning, scaling, and feature engineering
- **Model Evaluation**: Metrics for classification and regression
- **Overfitting/Underfitting**: Identification and mitigation strategies

### Practical Skills
- Implementing data preprocessing pipelines
- Building feature engineering utilities
- Creating model evaluation frameworks
- Developing complete ML applications
- Deploying models in production environments

### Real-World Applications
- Customer churn prediction
- Email spam detection
- House price prediction
- Medical diagnosis
- Recommendation systems

### Next Steps
In the following chapters, we'll dive deeper into specific machine learning algorithms:
- **Chapter 4**: Supervised Learning - Classification algorithms
- **Chapter 5**: Supervised Learning - Regression algorithms
- **Chapter 6**: Unsupervised Learning - Clustering and dimensionality reduction

The foundation we've built in this chapter will serve as the basis for understanding and implementing more advanced machine learning techniques throughout the rest of the book.

## Exercises

### Exercise 1: Data Preprocessing Pipeline
Create a comprehensive data preprocessing pipeline that handles:
- Missing value imputation
- Outlier detection and removal
- Feature scaling and normalization
- Categorical encoding
- Feature selection

### Exercise 2: Model Evaluation Framework
Implement a model evaluation framework that calculates:
- Classification metrics (accuracy, precision, recall, F1-score)
- Regression metrics (MSE, MAE, R²)
- Cross-validation results
- Learning curves

### Exercise 3: Customer Segmentation
Build an unsupervised learning system for customer segmentation:
- Implement K-means clustering
- Evaluate clustering quality
- Visualize customer segments
- Generate business insights

### Exercise 4: Feature Engineering Challenge
Create a feature engineering system for a real estate dataset:
- Generate polynomial features
- Create interaction features
- Implement domain-specific features
- Evaluate feature importance

### Exercise 5: End-to-End ML Project
Complete a full machine learning project from start to finish:
- Define the problem and success metrics
- Collect and explore data
- Preprocess and engineer features
- Train and evaluate multiple models
- Deploy the best model
- Monitor performance in production

These exercises will help you solidify your understanding of machine learning concepts and develop practical skills for building real-world AI applications with Java.

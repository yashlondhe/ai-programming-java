package com.aiprogramming.ch03;

import java.util.*;

/**
 * Chapter 3: Introduction to Machine Learning
 * 
 * This class demonstrates the fundamental concepts of machine learning
 * including data preprocessing, feature engineering, model training,
 * and evaluation using Java.
 */
public class MLBasicsDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 3: Introduction to Machine Learning ===\n");
        
        // Demonstrate data preprocessing
        demonstrateDataPreprocessing();
        
        // Demonstrate feature engineering
        demonstrateFeatureEngineering();
        
        // Demonstrate model evaluation
        demonstrateModelEvaluation();
        
        // Demonstrate overfitting and underfitting
        demonstrateOverfittingUnderfitting();
        
        // Demonstrate complete ML pipeline
        demonstrateCompletePipeline();
    }
    
    /**
     * Demonstrates various data preprocessing techniques
     */
    public static void demonstrateDataPreprocessing() {
        System.out.println("1. Data Preprocessing Demonstration");
        System.out.println("===================================");
        
        // Create sample dataset with missing values and outliers
        List<DataPoint> rawData = createSampleDataset();
        Dataset dataset = new Dataset(rawData);
        
        System.out.println("Original dataset size: " + dataset.size());
        System.out.println("Features: " + dataset.getFeatureNames());
        
        // Handle missing values
        MissingValueHandler missingHandler = new MissingValueHandler();
        Dataset dataWithoutMissing = missingHandler.fillWithMean(dataset, "age");
        System.out.println("After handling missing values: " + dataWithoutMissing.size());
        
        // Detect and handle outliers
        OutlierDetector outlierDetector = new OutlierDetector();
        List<DataPoint> outliers = outlierDetector.detectOutliersZScore(dataset, "income", 2.0);
        System.out.println("Detected outliers: " + outliers.size());
        
        // Scale features
        FeatureScaler scaler = new FeatureScaler();
        Dataset scaledData = scaler.standardize(dataWithoutMissing, "income");
        System.out.println("After scaling: " + scaledData.size());
        
        System.out.println();
    }
    
    /**
     * Demonstrates feature engineering techniques
     */
    public static void demonstrateFeatureEngineering() {
        System.out.println("2. Feature Engineering Demonstration");
        System.out.println("====================================");
        
        List<DataPoint> data = createSampleDataset();
        Dataset dataset = new Dataset(data);
        
        // Create polynomial features
        FeatureCreator featureCreator = new FeatureCreator();
        Dataset withPolynomialFeatures = featureCreator.createPolynomialFeatures(dataset, "age", 2);
        System.out.println("Added polynomial features for age");
        
        // Create interaction features
        Dataset withInteractionFeatures = featureCreator.createInteractionFeatures(dataset, "age", "income");
        System.out.println("Added interaction features between age and income");
        
        // Feature selection
        FeatureSelector selector = new FeatureSelector();
        List<String> selectedFeatures = selector.selectByCorrelation(dataset, "target", 0.1);
        System.out.println("Selected features by correlation: " + selectedFeatures);
        
        System.out.println();
    }
    
    /**
     * Demonstrates model evaluation techniques
     */
    public static void demonstrateModelEvaluation() {
        System.out.println("3. Model Evaluation Demonstration");
        System.out.println("=================================");
        
        // Create sample classification data
        List<DataPoint> classificationData = createClassificationDataset();
        Dataset dataset = new Dataset(classificationData);
        
        // Split data
        DatasetSplit split = dataset.split(0.7, 0.15, 0.15);
        
        // Train a simple model (simulated)
        SimpleClassifier classifier = new SimpleClassifier();
        TrainedModel model = classifier.train(split.getTrainingSet());
        
        // Evaluate on validation set
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        ClassificationMetrics metrics = evaluator.evaluate(model, split.getValidationSet());
        
        System.out.println("Classification Results:");
        System.out.printf("Accuracy: %.4f%n", metrics.getAccuracy());
        System.out.printf("Precision: %.4f%n", metrics.getPrecision());
        System.out.printf("Recall: %.4f%n", metrics.getRecall());
        System.out.printf("F1-Score: %.4f%n", metrics.getF1Score());
        
        // Cross-validation
        CrossValidator crossValidator = new CrossValidator();
        double cvScore = crossValidator.crossValidate(dataset, classifier, 5);
        System.out.printf("Cross-validation score: %.4f%n", cvScore);
        
        System.out.println();
    }
    
    /**
     * Demonstrates overfitting and underfitting concepts
     */
    public static void demonstrateOverfittingUnderfitting() {
        System.out.println("4. Overfitting and Underfitting Demonstration");
        System.out.println("=============================================");
        
        // Generate synthetic data
        List<DataPoint> trainingData = generateSyntheticData(20, 0.1);
        List<DataPoint> testData = generateSyntheticData(100, 0.1);
        
        Dataset trainingDataset = new Dataset(trainingData);
        Dataset testDataset = new Dataset(testData);
        
        // Test different model complexities
        for (int degree = 1; degree <= 5; degree++) {
            PolynomialRegression model = new PolynomialRegression(degree);
            model.train(trainingDataset);
            
            double trainingError = model.evaluate(trainingDataset);
            double testError = model.evaluate(testDataset);
            
            System.out.printf("Polynomial degree %d: Training Error=%.4f, Test Error=%.4f%n", 
                degree, trainingError, testError);
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrates a complete machine learning pipeline
     */
    public static void demonstrateCompletePipeline() {
        System.out.println("5. Complete ML Pipeline Demonstration");
        System.out.println("=====================================");
        
        // Load and preprocess data
        CustomerChurnAnalysis analysis = new CustomerChurnAnalysis();
        Dataset rawData = analysis.loadCustomerData("sample_customer_data.csv");
        
        ChurnDataPreprocessor preprocessor = new ChurnDataPreprocessor();
        Dataset processedData = preprocessor.preprocessData(rawData);
        
        // Split data
        DatasetSplit split = processedData.split(0.8, 0.1, 0.1);
        
        // Train multiple models
        List<MLAlgorithm> algorithms = Arrays.asList(
            new LogisticRegression(),
            new RandomForest(),
            new SupportVectorMachine()
        );
        
        Map<String, ClassificationMetrics> results = new HashMap<>();
        
        for (MLAlgorithm algorithm : algorithms) {
            String modelName = algorithm.getClass().getSimpleName();
            System.out.println("Training " + modelName + "...");
            
            TrainedModel model = algorithm.train(split.getTrainingSet());
            ClassificationMetrics metrics = new ClassificationEvaluator()
                .evaluate(model, split.getValidationSet());
            results.put(modelName, metrics);
            
            System.out.printf("%s - Accuracy: %.4f, F1: %.4f%n",
                modelName, metrics.getAccuracy(), metrics.getF1Score());
        }
        
        // Select best model
        String bestModel = results.entrySet().stream()
            .max(Comparator.comparing(entry -> entry.getValue().getF1Score()))
            .map(Map.Entry::getKey)
            .orElse("Unknown");
        
        System.out.println("Best model: " + bestModel);
        
        System.out.println();
    }
    
    /**
     * Creates a sample dataset for demonstration
     */
    private static List<DataPoint> createSampleDataset() {
        List<DataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 100; i++) {
            Map<String, Object> features = new HashMap<>();
            
            // Some missing values
            if (random.nextDouble() > 0.1) {
                features.put("age", 20 + random.nextInt(60));
            }
            
            features.put("income", 30000 + random.nextInt(100000));
            features.put("education", random.nextDouble() > 0.5 ? "college" : "high_school");
            features.put("location", random.nextDouble() > 0.5 ? "urban" : "suburban");
            
            // Target variable
            boolean target = random.nextDouble() > 0.6;
            
            data.add(new DataPoint(features, target));
        }
        
        return data;
    }
    
    /**
     * Creates a classification dataset
     */
    private static List<DataPoint> createClassificationDataset() {
        List<DataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 200; i++) {
            double x1 = random.nextDouble() * 4 - 2;
            double x2 = random.nextDouble() * 4 - 2;
            
            // Create a simple classification pattern
            int label = (x1 + x2 > 0) ? 1 : 0;
            
            Map<String, Object> features = new HashMap<>();
            features.put("feature1", x1);
            features.put("feature2", x2);
            
            data.add(new DataPoint(features, label));
        }
        
        return data;
    }
    
    /**
     * Generates synthetic data for overfitting demonstration
     */
    private static List<DataPoint> generateSyntheticData(int numPoints, double noiseLevel) {
        List<DataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numPoints; i++) {
            double x = random.nextDouble() * 10;
            double y = 2 * x + 1 + random.nextGaussian() * noiseLevel;
            
            Map<String, Object> features = new HashMap<>();
            features.put("x", x);
            
            data.add(new DataPoint(features, y));
        }
        
        return data;
    }
}

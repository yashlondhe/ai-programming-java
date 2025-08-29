package com.aiprogramming.ch04;

import com.aiprogramming.utils.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Chapter 4: Classification Algorithms
 * 
 * This class demonstrates various classification algorithms including
 * K-Nearest Neighbors, Naive Bayes, Decision Trees, and Logistic Regression.
 * Each algorithm is implemented from scratch to show the underlying principles.
 */
public class ClassificationDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 4: Classification Algorithms ===\n");
        
        // Demonstrate K-Nearest Neighbors
        demonstrateKNN();
        
        // Demonstrate Naive Bayes
        demonstrateNaiveBayes();
        
        // Demonstrate Decision Trees
        demonstrateDecisionTree();
        
        // Demonstrate Logistic Regression
        demonstrateLogisticRegression();
        
        // Demonstrate ensemble methods
        demonstrateEnsembleMethods();
        
        // Demonstrate complete classification pipeline
        demonstrateCompletePipeline();
    }
    
    /**
     * Demonstrates K-Nearest Neighbors classification
     */
    public static void demonstrateKNN() {
        System.out.println("1. K-Nearest Neighbors (KNN) Classification");
        System.out.println("===========================================");
        
        // Create sample dataset
        List<ClassificationDataPoint> trainingData = createSampleClassificationData();
        List<ClassificationDataPoint> testData = createTestData();
        
        // Train KNN classifier
        KNNClassifier knn = new KNNClassifier(3); // k=3
        knn.train(trainingData);
        
        System.out.println("Training data size: " + trainingData.size());
        System.out.println("Test data size: " + testData.size());
        
        // Make predictions
        int correct = 0;
        for (ClassificationDataPoint testPoint : testData) {
            String prediction = knn.predict(testPoint.getFeatures());
            String actual = testPoint.getLabel();
            
            if (prediction.equals(actual)) {
                correct++;
            }
            
            System.out.printf("Features: %s, Predicted: %s, Actual: %s%n", 
                testPoint.getFeatures(), prediction, actual);
        }
        
        double accuracy = (double) correct / testData.size();
        System.out.printf("KNN Accuracy: %.2f%%%n", accuracy * 100);
        
        // Using DataUtils for accuracy calculation
        double[] trueLabels = testData.stream()
            .mapToDouble(point -> point.getLabel().equals("positive") ? 1.0 : 0.0)
            .toArray();
        double[] predictedLabels = testData.stream()
            .mapToDouble(point -> knn.predict(point.getFeatures()).equals("positive") ? 1.0 : 0.0)
            .toArray();
        
        double accuracyFromUtils = DataUtils.accuracy(trueLabels, predictedLabels);
        System.out.printf("KNN Accuracy (using utils): %.2f%%%n", accuracyFromUtils * 100);
        System.out.println();
    }
    
    /**
     * Demonstrates Naive Bayes classification
     */
    public static void demonstrateNaiveBayes() {
        System.out.println("2. Naive Bayes Classification");
        System.out.println("=============================");
        
        // Create sample dataset
        List<ClassificationDataPoint> trainingData = createSampleClassificationData();
        List<ClassificationDataPoint> testData = createTestData();
        
        // Train Naive Bayes classifier
        NaiveBayesClassifier nb = new NaiveBayesClassifier();
        nb.train(trainingData);
        
        // Make predictions
        int correct = 0;
        for (ClassificationDataPoint testPoint : testData) {
            String prediction = nb.predict(testPoint.getFeatures());
            String actual = testPoint.getLabel();
            
            if (prediction.equals(actual)) {
                correct++;
            }
            
            System.out.printf("Features: %s, Predicted: %s, Actual: %s%n", 
                testPoint.getFeatures(), prediction, actual);
        }
        
        double accuracy = (double) correct / testData.size();
        System.out.printf("Naive Bayes Accuracy: %.2f%%%n", accuracy * 100);
        System.out.println();
    }
    
    /**
     * Demonstrates Decision Tree classification
     */
    public static void demonstrateDecisionTree() {
        System.out.println("3. Decision Tree Classification");
        System.out.println("===============================");
        
        // Using MatrixUtils for feature matrix operations
        System.out.println("Feature Matrix Operations with Utils:");
        double[][] featureMatrix = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        double[][] normalizedFeatures = DataUtils.normalize(featureMatrix);
        System.out.println("Normalized feature matrix:");
        DataUtils.printMatrix(normalizedFeatures, "Normalized Features");
        
        // Create sample dataset
        List<ClassificationDataPoint> trainingData = createSampleClassificationData();
        List<ClassificationDataPoint> testData = createTestData();
        
        // Train Decision Tree classifier
        DecisionTreeClassifier dt = new DecisionTreeClassifier(3); // max depth = 3
        dt.train(trainingData);
        
        // Make predictions
        int correct = 0;
        for (ClassificationDataPoint testPoint : testData) {
            String prediction = dt.predict(testPoint.getFeatures());
            String actual = testPoint.getLabel();
            
            if (prediction.equals(actual)) {
                correct++;
            }
            
            System.out.printf("Features: %s, Predicted: %s, Actual: %s%n", 
                testPoint.getFeatures(), prediction, actual);
        }
        
        double accuracy = (double) correct / testData.size();
        System.out.printf("Decision Tree Accuracy: %.2f%%%n", accuracy * 100);
        
        // Print tree structure
        System.out.println("Decision Tree Structure:");
        dt.printTree();
        System.out.println();
    }
    
    /**
     * Demonstrates Logistic Regression classification
     */
    public static void demonstrateLogisticRegression() {
        System.out.println("4. Logistic Regression Classification");
        System.out.println("====================================");
        
        // Create sample dataset
        List<ClassificationDataPoint> trainingData = createSampleClassificationData();
        List<ClassificationDataPoint> testData = createTestData();
        
        // Train Logistic Regression classifier
        LogisticRegressionClassifier lr = new LogisticRegressionClassifier(0.01, 1000);
        lr.train(trainingData);
        
        // Make predictions
        int correct = 0;
        for (ClassificationDataPoint testPoint : testData) {
            String prediction = lr.predict(testPoint.getFeatures());
            String actual = testPoint.getLabel();
            
            if (prediction.equals(actual)) {
                correct++;
            }
            
            System.out.printf("Features: %s, Predicted: %s, Actual: %s%n", 
                testPoint.getFeatures(), prediction, actual);
        }
        
        double accuracy = (double) correct / testData.size();
        System.out.printf("Logistic Regression Accuracy: %.2f%%%n", accuracy * 100);
        System.out.println();
    }
    
    /**
     * Demonstrates ensemble methods
     */
    public static void demonstrateEnsembleMethods() {
        System.out.println("5. Ensemble Methods");
        System.out.println("==================");
        
        // Create sample dataset
        List<ClassificationDataPoint> trainingData = createSampleClassificationData();
        List<ClassificationDataPoint> testData = createTestData();
        
        // Create ensemble classifier
        EnsembleClassifier ensemble = new EnsembleClassifier();
        ensemble.addClassifier(new KNNClassifier(3));
        ensemble.addClassifier(new NaiveBayesClassifier());
        ensemble.addClassifier(new DecisionTreeClassifier(3));
        
        ensemble.train(trainingData);
        
        // Make predictions
        int correct = 0;
        for (ClassificationDataPoint testPoint : testData) {
            String prediction = ensemble.predict(testPoint.getFeatures());
            String actual = testPoint.getLabel();
            
            if (prediction.equals(actual)) {
                correct++;
            }
            
            System.out.printf("Features: %s, Predicted: %s, Actual: %s%n", 
                testPoint.getFeatures(), prediction, actual);
        }
        
        double accuracy = (double) correct / testData.size();
        System.out.printf("Ensemble Accuracy: %.2f%%%n", accuracy * 100);
        System.out.println();
    }
    
    /**
     * Demonstrates complete classification pipeline
     */
    public static void demonstrateCompletePipeline() {
        System.out.println("6. Complete Classification Pipeline");
        System.out.println("===================================");
        
        // Create sample dataset
        List<ClassificationDataPoint> allData = createSampleClassificationData();
        allData.addAll(createTestData());
        
        // Split data
        ClassificationDataSplitter splitter = new ClassificationDataSplitter(0.8);
        ClassificationDataSplit split = splitter.split(allData);
        
        System.out.println("Training set size: " + split.getTrainingData().size());
        System.out.println("Test set size: " + split.getTestData().size());
        
        // Create and train multiple classifiers
        List<Classifier> classifiers = Arrays.asList(
            new KNNClassifier(3),
            new NaiveBayesClassifier(),
            new DecisionTreeClassifier(3),
            new LogisticRegressionClassifier(0.01, 1000)
        );
        
        // Evaluate each classifier
        ClassificationEvaluator evaluator = new ClassificationEvaluator();
        
        for (Classifier classifier : classifiers) {
            classifier.train(split.getTrainingData());
            
            List<String> predictions = new ArrayList<>();
            List<String> actuals = new ArrayList<>();
            
            for (ClassificationDataPoint testPoint : split.getTestData()) {
                predictions.add(classifier.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getLabel());
            }
            
            ClassificationMetrics metrics = evaluator.evaluate(actuals, predictions);
            
            System.out.printf("%s:%n", classifier.getClass().getSimpleName());
            System.out.printf("  Accuracy: %.2f%%%n", metrics.getAccuracy() * 100);
            System.out.printf("  Precision: %.2f%n", metrics.getPrecision());
            System.out.printf("  Recall: %.2f%n", metrics.getRecall());
            System.out.printf("  F1-Score: %.2f%n", metrics.getF1Score());
            System.out.println();
        }
    }
    
    /**
     * Creates sample classification data for demonstration
     */
    private static List<ClassificationDataPoint> createSampleClassificationData() {
        List<ClassificationDataPoint> data = new ArrayList<>();
        
        // Class 0: Low income, young age
        for (int i = 0; i < 20; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("income", 20000 + Math.random() * 10000);
            features.put("age", 20 + Math.random() * 10);
            data.add(new ClassificationDataPoint(features, "low_risk"));
        }
        
        // Class 1: High income, older age
        for (int i = 0; i < 20; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("income", 60000 + Math.random() * 20000);
            features.put("age", 40 + Math.random() * 20);
            data.add(new ClassificationDataPoint(features, "high_risk"));
        }
        
        // Class 2: Medium income, middle age
        for (int i = 0; i < 20; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("income", 35000 + Math.random() * 15000);
            features.put("age", 30 + Math.random() * 15);
            data.add(new ClassificationDataPoint(features, "medium_risk"));
        }
        
        return data;
    }
    
    /**
     * Creates test data for evaluation
     */
    private static List<ClassificationDataPoint> createTestData() {
        List<ClassificationDataPoint> data = new ArrayList<>();
        
        // Test cases
        Map<String, Double> test1 = new HashMap<>();
        test1.put("income", 25000.0);
        test1.put("age", 25.0);
        data.add(new ClassificationDataPoint(test1, "low_risk"));
        
        Map<String, Double> test2 = new HashMap<>();
        test2.put("income", 70000.0);
        test2.put("age", 45.0);
        data.add(new ClassificationDataPoint(test2, "high_risk"));
        
        Map<String, Double> test3 = new HashMap<>();
        test3.put("income", 40000.0);
        test3.put("age", 35.0);
        data.add(new ClassificationDataPoint(test3, "medium_risk"));
        
        return data;
    }
}

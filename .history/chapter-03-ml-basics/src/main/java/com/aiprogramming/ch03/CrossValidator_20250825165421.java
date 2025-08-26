package com.aiprogramming.ch03;

import java.util.*;

/**
 * Provides cross-validation functionality for machine learning models.
 */
public class CrossValidator {
    
    /**
     * Performs k-fold cross-validation
     */
    public double crossValidate(Dataset dataset, MLAlgorithm algorithm, int folds) {
        List<Double> foldScores = new ArrayList<>();
        List<Dataset> foldDatasets = createFolds(dataset, folds);
        
        for (int i = 0; i < folds; i++) {
            // Create training and validation sets
            Dataset validationSet = foldDatasets.get(i);
            Dataset trainingSet = combineFolds(foldDatasets, i);
            
            // Train model
            TrainedModel model = algorithm.train(trainingSet);
            
            // Evaluate on validation set
            double score = model.evaluate(validationSet);
            foldScores.add(score);
        }
        
        return foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Performs stratified k-fold cross-validation for classification
     */
    public double stratifiedCrossValidate(Dataset dataset, MLAlgorithm algorithm, int folds) {
        List<Double> foldScores = new ArrayList<>();
        List<Dataset> foldDatasets = createStratifiedFolds(dataset, folds);
        
        for (int i = 0; i < folds; i++) {
            Dataset validationSet = foldDatasets.get(i);
            Dataset trainingSet = combineFolds(foldDatasets, i);
            
            TrainedModel model = algorithm.train(trainingSet);
            double score = model.evaluate(validationSet);
            foldScores.add(score);
        }
        
        return foldScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Performs leave-one-out cross-validation
     */
    public double leaveOneOutCrossValidate(Dataset dataset, MLAlgorithm algorithm) {
        List<Double> scores = new ArrayList<>();
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        for (int i = 0; i < dataPoints.size(); i++) {
            // Create training set (all except one)
            List<DataPoint> trainingData = new ArrayList<>(dataPoints);
            DataPoint testPoint = trainingData.remove(i);
            
            Dataset trainingSet = new Dataset(trainingData);
            Dataset testSet = new Dataset(List.of(testPoint));
            
            // Train and evaluate
            TrainedModel model = algorithm.train(trainingSet);
            double score = model.evaluate(testSet);
            scores.add(score);
        }
        
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Performs time series cross-validation
     */
    public double timeSeriesCrossValidate(Dataset dataset, MLAlgorithm algorithm, int trainSize, int testSize) {
        List<Double> scores = new ArrayList<>();
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        for (int i = trainSize; i <= dataPoints.size() - testSize; i += testSize) {
            // Training set: from start to i
            List<DataPoint> trainingData = dataPoints.subList(0, i);
            // Test set: from i to i + testSize
            List<DataPoint> testData = dataPoints.subList(i, Math.min(i + testSize, dataPoints.size()));
            
            Dataset trainingSet = new Dataset(trainingData);
            Dataset testSet = new Dataset(testData);
            
            TrainedModel model = algorithm.train(trainingSet);
            double score = model.evaluate(testSet);
            scores.add(score);
        }
        
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Performs nested cross-validation for hyperparameter tuning
     */
    public Map<String, Object> nestedCrossValidate(Dataset dataset, MLAlgorithm algorithm, 
                                                  List<Map<String, Object>> hyperparameterSets, int outerFolds, int innerFolds) {
        List<Dataset> outerFoldsData = createFolds(dataset, outerFolds);
        Map<String, Object> bestHyperparameters = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (Map<String, Object> hyperparams : hyperparameterSets) {
            List<Double> outerScores = new ArrayList<>();
            
            for (int i = 0; i < outerFolds; i++) {
                Dataset outerValidationSet = outerFoldsData.get(i);
                Dataset outerTrainingSet = combineFolds(outerFoldsData, i);
                
                // Inner cross-validation for hyperparameter selection
                List<Dataset> innerFoldsData = createFolds(outerTrainingSet, innerFolds);
                double innerScore = 0.0;
                
                for (int j = 0; j < innerFolds; j++) {
                    Dataset innerValidationSet = innerFoldsData.get(j);
                    Dataset innerTrainingSet = combineFolds(innerFoldsData, j);
                    
                    // Set hyperparameters and train
                    algorithm.setHyperparameters(hyperparams);
                    TrainedModel model = algorithm.train(innerTrainingSet);
                    innerScore += model.evaluate(innerValidationSet);
                }
                
                innerScore /= innerFolds;
                
                // Train on full outer training set with best hyperparameters
                algorithm.setHyperparameters(hyperparams);
                TrainedModel model = algorithm.train(outerTrainingSet);
                double outerScore = model.evaluate(outerValidationSet);
                outerScores.add(outerScore);
            }
            
            double avgOuterScore = outerScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            
            if (avgOuterScore > bestScore) {
                bestScore = avgOuterScore;
                bestHyperparameters = new HashMap<>(hyperparams);
            }
        }
        
        return bestHyperparameters;
    }
    
    /**
     * Creates k folds from a dataset
     */
    private List<Dataset> createFolds(Dataset dataset, int folds) {
        List<DataPoint> dataPoints = new ArrayList<>(dataset.getDataPoints());
        Collections.shuffle(dataPoints, new Random(42)); // For reproducibility
        
        List<Dataset> foldDatasets = new ArrayList<>();
        int foldSize = dataPoints.size() / folds;
        
        for (int i = 0; i < folds; i++) {
            int startIndex = i * foldSize;
            int endIndex = (i == folds - 1) ? dataPoints.size() : (i + 1) * foldSize;
            
            List<DataPoint> foldData = dataPoints.subList(startIndex, endIndex);
            foldDatasets.add(new Dataset(foldData));
        }
        
        return foldDatasets;
    }
    
    /**
     * Creates stratified k folds for classification
     */
    private List<Dataset> createStratifiedFolds(Dataset dataset, int folds) {
        // Group data points by target class
        Map<Object, List<DataPoint>> classGroups = new HashMap<>();
        for (DataPoint point : dataset.getDataPoints()) {
            Object target = point.getTarget();
            classGroups.computeIfAbsent(target, k -> new ArrayList<>()).add(point);
        }
        
        List<Dataset> foldDatasets = new ArrayList<>();
        for (int i = 0; i < folds; i++) {
            foldDatasets.add(new Dataset());
        }
        
        // Distribute each class across folds
        for (List<DataPoint> classData : classGroups.values()) {
            Collections.shuffle(classData, new Random(42));
            
            for (int i = 0; i < classData.size(); i++) {
                int foldIndex = i % folds;
                foldDatasets.get(foldIndex).addDataPoint(classData.get(i));
            }
        }
        
        return foldDatasets;
    }
    
    /**
     * Combines all folds except the specified one
     */
    private Dataset combineFolds(List<Dataset> foldDatasets, int excludeIndex) {
        List<DataPoint> combinedData = new ArrayList<>();
        
        for (int i = 0; i < foldDatasets.size(); i++) {
            if (i != excludeIndex) {
                combinedData.addAll(foldDatasets.get(i).getDataPoints());
            }
        }
        
        return new Dataset(combinedData);
    }
    
    /**
     * Generates cross-validation report
     */
    public void generateCrossValidationReport(Dataset dataset, MLAlgorithm algorithm, int folds) {
        System.out.println("=== Cross-Validation Report ===");
        
        // Standard k-fold CV
        double kFoldScore = crossValidate(dataset, algorithm, folds);
        System.out.printf("%d-fold cross-validation score: %.4f%n", folds, kFoldScore);
        
        // Stratified k-fold CV
        double stratifiedScore = stratifiedCrossValidate(dataset, algorithm, folds);
        System.out.printf("Stratified %d-fold cross-validation score: %.4f%n", folds, stratifiedScore);
        
        // Leave-one-out CV
        double looScore = leaveOneOutCrossValidate(dataset, algorithm);
        System.out.printf("Leave-one-out cross-validation score: %.4f%n", looScore);
        
        // Time series CV
        int trainSize = dataset.size() / 3;
        int testSize = dataset.size() / 10;
        double timeSeriesScore = timeSeriesCrossValidate(dataset, algorithm, trainSize, testSize);
        System.out.printf("Time series cross-validation score: %.4f%n", timeSeriesScore);
    }
}

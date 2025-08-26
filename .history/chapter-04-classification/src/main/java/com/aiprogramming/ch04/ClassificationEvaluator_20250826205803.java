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
    
    /**
     * Calculates overall accuracy
     */
    private double calculateAccuracy(Map<String, Map<String, Integer>> confusionMatrix) {
        int correct = 0;
        int total = 0;
        
        for (String actualClass : confusionMatrix.keySet()) {
            for (String predictedClass : confusionMatrix.get(actualClass).keySet()) {
                int count = confusionMatrix.get(actualClass).get(predictedClass);
                if (actualClass.equals(predictedClass)) {
                    correct += count;
                }
                total += count;
            }
        }
        
        return total > 0 ? (double) correct / total : 0.0;
    }
    
    /**
     * Calculates macro-averaged precision
     */
    private double calculatePrecision(Map<String, Map<String, Integer>> confusionMatrix) {
        double totalPrecision = 0.0;
        int numClasses = confusionMatrix.size();
        
        for (String predictedClass : confusionMatrix.keySet()) {
            int truePositives = confusionMatrix.get(predictedClass).getOrDefault(predictedClass, 0);
            int falsePositives = 0;
            
            for (String actualClass : confusionMatrix.keySet()) {
                if (!actualClass.equals(predictedClass)) {
                    falsePositives += confusionMatrix.get(actualClass).getOrDefault(predictedClass, 0);
                }
            }
            
            double precision = (truePositives + falsePositives) > 0 ? 
                (double) truePositives / (truePositives + falsePositives) : 0.0;
            totalPrecision += precision;
        }
        
        return numClasses > 0 ? totalPrecision / numClasses : 0.0;
    }
    
    /**
     * Calculates macro-averaged recall
     */
    private double calculateRecall(Map<String, Map<String, Integer>> confusionMatrix) {
        double totalRecall = 0.0;
        int numClasses = confusionMatrix.size();
        
        for (String actualClass : confusionMatrix.keySet()) {
            int truePositives = confusionMatrix.get(actualClass).getOrDefault(actualClass, 0);
            int falseNegatives = 0;
            
            for (String predictedClass : confusionMatrix.keySet()) {
                if (!predictedClass.equals(actualClass)) {
                    falseNegatives += confusionMatrix.get(actualClass).getOrDefault(predictedClass, 0);
                }
            }
            
            double recall = (truePositives + falseNegatives) > 0 ? 
                (double) truePositives / (truePositives + falseNegatives) : 0.0;
            totalRecall += recall;
        }
        
        return numClasses > 0 ? totalRecall / numClasses : 0.0;
    }
    
    /**
     * Calculates F1 score
     */
    private double calculateF1Score(double precision, double recall) {
        return (precision + recall) > 0 ? 
            2 * precision * recall / (precision + recall) : 0.0;
    }
}

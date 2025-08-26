package com.aiprogramming.ch03;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Provides methods for evaluating classification models.
 */
public class ClassificationEvaluator {
    
    /**
     * Evaluates a trained model on a test dataset
     */
    public ClassificationMetrics evaluate(TrainedModel model, Dataset testData) {
        List<Prediction> predictions = model.predict(testData);
        List<Object> actualLabels = testData.getLabels();
        
        // Calculate confusion matrix
        Map<String, Map<String, Integer>> confusionMatrix = calculateConfusionMatrix(predictions, actualLabels);
        
        // Calculate metrics
        double accuracy = calculateAccuracy(confusionMatrix);
        double precision = calculatePrecision(confusionMatrix);
        double recall = calculateRecall(confusionMatrix);
        double f1Score = calculateF1Score(precision, recall);
        
        return new ClassificationMetrics(accuracy, precision, recall, f1Score, confusionMatrix);
    }
    
    /**
     * Calculates the confusion matrix
     */
    private Map<String, Map<String, Integer>> calculateConfusionMatrix(List<Prediction> predictions, List<Object> actualLabels) {
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            String predicted = predictions.get(i).getPredictedLabel().toString();
            String actual = actualLabels.get(i).toString();
            
            confusionMatrix.computeIfAbsent(predicted, k -> new HashMap<>());
            confusionMatrix.get(predicted).merge(actual, 1, Integer::sum);
        }
        
        return confusionMatrix;
    }
    
    /**
     * Calculates accuracy from confusion matrix
     */
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
        
        return total > 0 ? (double) correct / total : 0.0;
    }
    
    /**
     * Calculates precision from confusion matrix
     */
    private double calculatePrecision(Map<String, Map<String, Integer>> confusionMatrix) {
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
        
        return numClasses > 0 ? totalPrecision / numClasses : 0.0;
    }
    
    /**
     * Calculates recall from confusion matrix
     */
    private double calculateRecall(Map<String, Map<String, Integer>> confusionMatrix) {
        double totalRecall = 0.0;
        int numClasses = confusionMatrix.size();
        
        for (String actual : confusionMatrix.keySet()) {
            int truePositives = 0;
            int falseNegatives = 0;
            
            for (String predicted : confusionMatrix.keySet()) {
                int count = confusionMatrix.get(predicted).getOrDefault(actual, 0);
                if (predicted.equals(actual)) {
                    truePositives = count;
                } else {
                    falseNegatives += count;
                }
            }
            
            double recall = (truePositives + falseNegatives) > 0 ? 
                (double) truePositives / (truePositives + falseNegatives) : 0.0;
            totalRecall += recall;
        }
        
        return numClasses > 0 ? totalRecall / numClasses : 0.0;
    }
    
    /**
     * Calculates F1-score from precision and recall
     */
    private double calculateF1Score(double precision, double recall) {
        return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
    }
    
    /**
     * Calculates per-class metrics
     */
    public Map<String, ClassMetrics> calculatePerClassMetrics(TrainedModel model, Dataset testData) {
        List<Prediction> predictions = model.predict(testData);
        List<Object> actualLabels = testData.getLabels();
        
        Map<String, Map<String, Integer>> confusionMatrix = calculateConfusionMatrix(predictions, actualLabels);
        Map<String, ClassMetrics> perClassMetrics = new HashMap<>();
        
        for (String className : confusionMatrix.keySet()) {
            int truePositives = confusionMatrix.get(className).getOrDefault(className, 0);
            int falsePositives = 0;
            int falseNegatives = 0;
            
            // Calculate false positives
            for (String predicted : confusionMatrix.keySet()) {
                if (!predicted.equals(className)) {
                    falsePositives += confusionMatrix.get(predicted).getOrDefault(className, 0);
                }
            }
            
            // Calculate false negatives
            for (String actual : confusionMatrix.keySet()) {
                if (!actual.equals(className)) {
                    falseNegatives += confusionMatrix.get(className).getOrDefault(actual, 0);
                }
            }
            
            double precision = (truePositives + falsePositives) > 0 ? 
                (double) truePositives / (truePositives + falsePositives) : 0.0;
            double recall = (truePositives + falseNegatives) > 0 ? 
                (double) truePositives / (truePositives + falseNegatives) : 0.0;
            double f1Score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
            
            perClassMetrics.put(className, new ClassMetrics(precision, recall, f1Score));
        }
        
        return perClassMetrics;
    }
    
    /**
     * Calculates ROC curve points
     */
    public List<ROCPoint> calculateROCCurve(TrainedModel model, Dataset testData) {
        List<Prediction> predictions = model.predict(testData);
        List<ROCPoint> rocPoints = new ArrayList<>();
        
        // Sort predictions by probability in descending order
        predictions.sort((p1, p2) -> Double.compare(p2.getProbability(), p1.getProbability()));
        
        int totalPositives = (int) predictions.stream()
                .filter(p -> Boolean.TRUE.equals(p.getActualLabel()))
                .count();
        int totalNegatives = predictions.size() - totalPositives;
        
        int truePositives = 0;
        int falsePositives = 0;
        
        for (Prediction prediction : predictions) {
            if (Boolean.TRUE.equals(prediction.getActualLabel())) {
                truePositives++;
            } else {
                falsePositives++;
            }
            
            double tpr = totalPositives > 0 ? (double) truePositives / totalPositives : 0.0;
            double fpr = totalNegatives > 0 ? (double) falsePositives / totalNegatives : 0.0;
            
            rocPoints.add(new ROCPoint(fpr, tpr, prediction.getProbability()));
        }
        
        return rocPoints;
    }
    
    /**
     * Calculates AUC (Area Under Curve) for ROC
     */
    public double calculateAUC(List<ROCPoint> rocPoints) {
        double auc = 0.0;
        
        for (int i = 1; i < rocPoints.size(); i++) {
            ROCPoint prev = rocPoints.get(i - 1);
            ROCPoint curr = rocPoints.get(i);
            
            // Trapezoidal rule
            double width = curr.getFPR() - prev.getFPR();
            double height = (prev.getTPR() + curr.getTPR()) / 2.0;
            auc += width * height;
        }
        
        return auc;
    }
    
    /**
     * Class for storing per-class metrics
     */
    public static class ClassMetrics {
        private final double precision;
        private final double recall;
        private final double f1Score;
        
        public ClassMetrics(double precision, double recall, double f1Score) {
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
        }
        
        public double getPrecision() { return precision; }
        public double getRecall() { return recall; }
        public double getF1Score() { return f1Score; }
    }
    
    /**
     * Class for storing ROC curve points
     */
    public static class ROCPoint {
        private final double fpr; // False Positive Rate
        private final double tpr; // True Positive Rate
        private final double threshold;
        
        public ROCPoint(double fpr, double tpr, double threshold) {
            this.fpr = fpr;
            this.tpr = tpr;
            this.threshold = threshold;
        }
        
        public double getFPR() { return fpr; }
        public double getTPR() { return tpr; }
        public double getThreshold() { return threshold; }
    }
}

/**
 * Classification metrics container
 */
class ClassificationMetrics {
    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;
    private final Map<String, Map<String, Integer>> confusionMatrix;
    
    public ClassificationMetrics(double accuracy, double precision, double recall, double f1Score,
                               Map<String, Map<String, Integer>> confusionMatrix) {
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.confusionMatrix = confusionMatrix;
    }
    
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public Map<String, Map<String, Integer>> getConfusionMatrix() { return confusionMatrix; }
}

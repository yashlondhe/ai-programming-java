package com.aiprogramming.ch04;

import java.util.Map;

/**
 * Holds classification evaluation metrics.
 */
public class ClassificationMetrics {
    
    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;
    private final Map<String, Map<String, Integer>> confusionMatrix;
    
    public ClassificationMetrics(double accuracy, double precision, double recall, 
                               double f1Score, Map<String, Map<String, Integer>> confusionMatrix) {
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.confusionMatrix = confusionMatrix;
    }
    
    /**
     * Gets the accuracy
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Gets the precision
     */
    public double getPrecision() {
        return precision;
    }
    
    /**
     * Gets the recall
     */
    public double getRecall() {
        return recall;
    }
    
    /**
     * Gets the F1 score
     */
    public double getF1Score() {
        return f1Score;
    }
    
    /**
     * Gets the confusion matrix
     */
    public Map<String, Map<String, Integer>> getConfusionMatrix() {
        return confusionMatrix;
    }
    
    @Override
    public String toString() {
        return String.format("ClassificationMetrics{accuracy=%.4f, precision=%.4f, recall=%.4f, f1Score=%.4f}", 
                           accuracy, precision, recall, f1Score);
    }
}

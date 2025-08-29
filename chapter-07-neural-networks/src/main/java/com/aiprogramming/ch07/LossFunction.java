package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;

/**
 * Loss function interface
 */
public interface LossFunction {
    double compute(List<Double> predictions, List<Double> targets);
    List<Double> derivative(List<Double> predictions, List<Double> targets);
}

/**
 * Mean Squared Error loss function
 */
class MeanSquaredError implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double diff = predictions.get(i) - targets.get(i);
            sum += diff * diff;
        }
        
        return sum / predictions.size();
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double derivative = 2.0 * (predictions.get(i) - targets.get(i)) / predictions.size();
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}

/**
 * Cross-entropy loss function for classification
 */
class CrossEntropy implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            sum += -targets.get(i) * Math.log(pred) - (1.0 - targets.get(i)) * Math.log(1.0 - pred);
        }
        
        return sum / predictions.size();
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            double derivative = (pred - targets.get(i)) / (pred * (1.0 - pred)) / predictions.size();
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}

/**
 * Categorical cross-entropy for multi-class classification
 */
class CategoricalCrossEntropy implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            if (targets.get(i) > 0) { // Only compute loss for non-zero targets
                sum += -targets.get(i) * Math.log(pred);
            }
        }
        
        return sum;
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            double derivative = -targets.get(i) / pred;
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}

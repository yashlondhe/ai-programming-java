package com.aiprogramming.ch05;

import java.util.*;

/**
 * Comprehensive regression evaluation metrics
 */
public class RegressionEvaluator {
    
    public RegressionMetrics evaluate(List<Double> actualValues, List<Double> predictedValues) {
        if (actualValues.size() != predictedValues.size()) {
            throw new IllegalArgumentException("Actual and predicted values must have the same size");
        }
        
        if (actualValues.isEmpty()) {
            throw new IllegalArgumentException("Cannot evaluate with empty lists");
        }
        
        int n = actualValues.size();
        
        // Calculate Mean Absolute Error (MAE)
        double mae = calculateMAE(actualValues, predictedValues);
        
        // Calculate Mean Squared Error (MSE)
        double mse = calculateMSE(actualValues, predictedValues);
        
        // Calculate Root Mean Squared Error (RMSE)
        double rmse = Math.sqrt(mse);
        
        // Calculate R² (Coefficient of Determination)
        double r2 = calculateR2(actualValues, predictedValues);
        
        // Calculate Mean Absolute Percentage Error (MAPE)
        double mape = calculateMAPE(actualValues, predictedValues);
        
        // Calculate residuals
        List<Double> residuals = calculateResiduals(actualValues, predictedValues);
        
        return new RegressionMetrics(mae, mse, rmse, r2, mape, residuals);
    }
    
    private double calculateMAE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            sum += Math.abs(actual.get(i) - predicted.get(i));
        }
        return sum / actual.size();
    }
    
    private double calculateMSE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            double error = actual.get(i) - predicted.get(i);
            sum += error * error;
        }
        return sum / actual.size();
    }
    
    private double calculateR2(List<Double> actual, List<Double> predicted) {
        // Calculate mean of actual values
        double meanActual = actual.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        // Calculate total sum of squares (TSS)
        double tss = 0.0;
        for (double value : actual) {
            double diff = value - meanActual;
            tss += diff * diff;
        }
        
        // Calculate residual sum of squares (RSS)
        double rss = 0.0;
        for (int i = 0; i < actual.size(); i++) {
            double residual = actual.get(i) - predicted.get(i);
            rss += residual * residual;
        }
        
        // R² = 1 - (RSS / TSS)
        return (tss == 0) ? 0.0 : 1.0 - (rss / tss);
    }
    
    private double calculateMAPE(List<Double> actual, List<Double> predicted) {
        double sum = 0.0;
        int validCount = 0;
        
        for (int i = 0; i < actual.size(); i++) {
            double actualValue = actual.get(i);
            if (Math.abs(actualValue) > 1e-10) { // Avoid division by zero
                sum += Math.abs((actualValue - predicted.get(i)) / actualValue);
                validCount++;
            }
        }
        
        return validCount > 0 ? (sum / validCount) * 100.0 : 0.0;
    }
    
    private List<Double> calculateResiduals(List<Double> actual, List<Double> predicted) {
        List<Double> residuals = new ArrayList<>();
        for (int i = 0; i < actual.size(); i++) {
            residuals.add(actual.get(i) - predicted.get(i));
        }
        return residuals;
    }
}
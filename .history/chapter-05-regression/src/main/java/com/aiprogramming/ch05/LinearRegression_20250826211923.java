package com.aiprogramming.ch05;

import java.util.*;
import org.apache.commons.math3.linear.*;

/**
 * Linear Regression Implementation using Normal Equation with robust matrix operations
 */
public class LinearRegression implements Regressor {
    
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public LinearRegression() {
        this.coefficients = new HashMap<>();
        this.intercept = 0.0;
        this.featureNames = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Convert data to matrix format
        int numSamples = trainingData.size();
        int numFeatures = featureNames.size();
        
        // X matrix (with bias column)
        double[][] X = new double[numSamples][numFeatures + 1];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            X[i][0] = 1.0; // Bias term
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j + 1] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        try {
            // Use Apache Commons Math for robust matrix operations
            RealMatrix XMatrix = MatrixUtils.createRealMatrix(X);
            RealVector yVector = MatrixUtils.createRealVector(y);
            
            // Solve normal equation: β = (X'X)^(-1)X'y
            RealMatrix XTranspose = XMatrix.transpose();
            RealMatrix XTX = XTranspose.multiply(XMatrix);
            RealVector XTy = XTranspose.operate(yVector);
            
            // Use QR decomposition for more stable solution
            QRDecomposition qr = new QRDecomposition(XMatrix);
            RealVector beta = qr.getSolver().solve(yVector);
            
            // Extract intercept and coefficients
            this.intercept = beta.getEntry(0);
            this.coefficients.clear();
            
            for (int i = 0; i < numFeatures; i++) {
                coefficients.put(featureNames.get(i), beta.getEntry(i + 1));
            }
            
            this.trained = true;
            
        } catch (Exception e) {
            // Fallback to simple mean if matrix operations fail
            System.err.println("Warning: Matrix operations failed, using fallback method: " + e.getMessage());
            fallbackTraining(trainingData);
        }
    }
    
    /**
     * Fallback training method using simple statistics
     */
    private void fallbackTraining(List<RegressionDataPoint> trainingData) {
        this.coefficients.clear();
        this.intercept = 0.0;
        
        // Calculate mean target
        double meanTarget = trainingData.stream()
                .mapToDouble(RegressionDataPoint::getTarget)
                .average()
                .orElse(0.0);
        
        this.intercept = meanTarget;
        
        // Simple linear coefficients based on correlation
        for (String featureName : featureNames) {
            double coefficient = calculateSimpleCoefficient(trainingData, featureName);
            coefficients.put(featureName, coefficient);
        }
        
        this.trained = true;
    }
    
    /**
     * Calculate simple coefficient based on correlation
     */
    private double calculateSimpleCoefficient(List<RegressionDataPoint> data, String featureName) {
        double meanFeature = data.stream()
                .mapToDouble(point -> point.getFeatures().getOrDefault(featureName, 0.0))
                .average()
                .orElse(0.0);
        
        double meanTarget = data.stream()
                .mapToDouble(RegressionDataPoint::getTarget)
                .average()
                .orElse(0.0);
        
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (RegressionDataPoint point : data) {
            double featureValue = point.getFeatures().getOrDefault(featureName, 0.0);
            double targetValue = point.getTarget();
            
            double featureDiff = featureValue - meanFeature;
            double targetDiff = targetValue - meanTarget;
            
            numerator += featureDiff * targetDiff;
            denominator += featureDiff * featureDiff;
        }
        
        return denominator > 1e-10 ? numerator / denominator : 0.0;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double prediction = intercept;
        
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            double coefficient = coefficients.getOrDefault(featureName, 0.0);
            prediction += coefficient * featureValue;
        }
        
        return prediction;
    }
    
    /**
     * Gets the learned coefficients
     */
    public Map<String, Double> getCoefficients() {
        return new HashMap<>(coefficients);
    }
    
    /**
     * Gets the learned intercept
     */
    public double getIntercept() {
        return intercept;
    }
    
    /**
     * Gets the feature importance based on absolute coefficient values
     */
    public Map<String, Double> getFeatureImportance() {
        Map<String, Double> importance = new HashMap<>();
        double maxCoeff = coefficients.values().stream()
                .mapToDouble(Math::abs)
                .max()
                .orElse(1.0);
        
        for (Map.Entry<String, Double> entry : coefficients.entrySet()) {
            importance.put(entry.getKey(), Math.abs(entry.getValue()) / maxCoeff);
        }
        
        return importance;
    }
    

}
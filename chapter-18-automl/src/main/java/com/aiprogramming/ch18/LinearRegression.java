package com.aiprogramming.ch18;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple Linear Regression model
 */
public class LinearRegression implements MLModel {
    
    private double[] coefficients;
    private double intercept;
    private Map<String, Object> hyperparameters;
    
    public LinearRegression() {
        this.hyperparameters = new HashMap<>();
        this.hyperparameters.put("learningRate", 0.01);
        this.hyperparameters.put("maxIterations", 1000);
        this.hyperparameters.put("regularization", 0.0);
    }
    
    @Override
    public void train(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        int numSamples = features.length;
        
        // Initialize coefficients
        coefficients = new double[numFeatures];
        intercept = 0.0;
        
        double learningRate = (Double) hyperparameters.get("learningRate");
        int maxIterations = (Integer) hyperparameters.get("maxIterations");
        double regularization = (Double) hyperparameters.get("regularization");
        
        // Gradient descent
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[] gradients = new double[numFeatures];
            double interceptGradient = 0.0;
            
            // Calculate gradients
            for (int i = 0; i < numSamples; i++) {
                double prediction = predict(features[i]);
                double error = prediction - targets[i];
                
                // Feature gradients
                for (int j = 0; j < numFeatures; j++) {
                    gradients[j] += error * features[i][j];
                }
                
                // Intercept gradient
                interceptGradient += error;
            }
            
            // Update parameters
            for (int j = 0; j < numFeatures; j++) {
                double gradient = gradients[j] / numSamples;
                if (regularization > 0) {
                    gradient += regularization * coefficients[j];
                }
                coefficients[j] -= learningRate * gradient;
            }
            
            intercept -= learningRate * (interceptGradient / numSamples);
        }
    }
    
    @Override
    public double[] predict(double[][] features) {
        double[] predictions = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            predictions[i] = predict(features[i]);
        }
        return predictions;
    }
    
    private double predict(double[] features) {
        double prediction = intercept;
        for (int i = 0; i < features.length; i++) {
            prediction += coefficients[i] * features[i];
        }
        return prediction;
    }
    
    @Override
    public double evaluate(double[][] features, double[] targets) {
        double[] predictions = predict(features);
        double mse = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - targets[i];
            mse += error * error;
        }
        return mse / predictions.length;
    }
    
    @Override
    public void setHyperparameters(Map<String, Object> hyperparameters) {
        this.hyperparameters.putAll(hyperparameters);
    }
    
    @Override
    public Map<String, Object> getHyperparameters() {
        return new HashMap<>(hyperparameters);
    }
    
    public double[] getCoefficients() {
        return coefficients;
    }
    
    public double getIntercept() {
        return intercept;
    }
}

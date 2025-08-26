package com.aiprogramming.ch04;

import java.util.*;

/**
 * Logistic Regression classifier implementation.
 * Uses gradient descent to train a binary classifier.
 */
public class LogisticRegressionClassifier implements Classifier {
    
    private final double learningRate;
    private final int maxIterations;
    private Map<String, Double> weights;
    private double bias;
    private boolean isTrained = false;
    
    public LogisticRegressionClassifier(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.weights = new HashMap<>();
        this.bias = 0.0;
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Initialize weights for all features
        Set<String> allFeatures = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            allFeatures.addAll(point.getFeatures().keySet());
        }
        
        for (String feature : allFeatures) {
            weights.putIfAbsent(feature, 0.0);
        }
        
        // Convert labels to binary (assuming first class is positive)
        Set<String> uniqueLabels = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            uniqueLabels.add(point.getLabel());
        }
        String positiveClass = uniqueLabels.iterator().next();
        
        // Gradient descent
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double totalLoss = 0.0;
            Map<String, Double> weightGradients = new HashMap<>();
            double biasGradient = 0.0;
            
            // Initialize gradients
            for (String feature : weights.keySet()) {
                weightGradients.put(feature, 0.0);
            }
            
            // Calculate gradients for each training example
            for (ClassificationDataPoint point : trainingData) {
                double prediction = predictProbability(point.getFeatures());
                int actual = point.getLabel().equals(positiveClass) ? 1 : 0;
                
                // Calculate loss
                double loss = -actual * Math.log(prediction + 1e-15) - 
                             (1 - actual) * Math.log(1 - prediction + 1e-15);
                totalLoss += loss;
                
                // Calculate gradients
                double error = prediction - actual;
                biasGradient += error;
                
                for (String feature : weights.keySet()) {
                    double featureValue = point.getFeature(feature);
                    weightGradients.put(feature, weightGradients.get(feature) + error * featureValue);
                }
            }
            
            // Update weights and bias
            bias -= learningRate * biasGradient / trainingData.size();
            
            for (String feature : weights.keySet()) {
                double gradient = weightGradients.get(feature) / trainingData.size();
                weights.put(feature, weights.get(feature) - learningRate * gradient);
            }
            
            // Early stopping if loss is very small
            if (totalLoss / trainingData.size() < 0.01) {
                break;
            }
        }
        
        isTrained = true;
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        double probability = predictProbability(features);
        return probability >= 0.5 ? "positive" : "negative";
    }
    
    /**
     * Predicts the probability of the positive class
     */
    private double predictProbability(Map<String, Double> features) {
        double z = bias;
        
        for (String feature : weights.keySet()) {
            double featureValue = features.getOrDefault(feature, 0.0);
            z += weights.get(feature) * featureValue;
        }
        
        return sigmoid(z);
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    /**
     * Gets the trained weights
     */
    public Map<String, Double> getWeights() {
        return new HashMap<>(weights);
    }
    
    /**
     * Gets the trained bias
     */
    public double getBias() {
        return bias;
    }
}

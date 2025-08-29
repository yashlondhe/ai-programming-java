package com.aiprogramming.ch17.utils;

import java.util.Random;

/**
 * Model wrapper that provides a unified interface for different ML models
 * 
 * This class simulates a machine learning model for demonstration purposes.
 * In a real implementation, this would wrap actual ML models from libraries
 * like Weka, DL4J, or other Java ML frameworks.
 */
public class ModelWrapper {
    
    private double[] weights;
    private double bias;
    private boolean isTrained;
    private Random random;
    
    /**
     * Constructor for model wrapper
     */
    public ModelWrapper() {
        this.random = new Random(42);
        this.isTrained = false;
    }
    
    /**
     * Train the model on the provided data
     * 
     * @param trainingData training dataset
     */
    public void trainModel(double[][] trainingData) {
        if (trainingData == null || trainingData.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        
        int numFeatures = trainingData[0].length;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        
        // Simple linear model training simulation
        // In a real implementation, this would use actual ML algorithms
        
        // Initialize weights randomly
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = random.nextGaussian() * 0.1;
        }
        
        // Simple gradient descent simulation
        double learningRate = 0.01;
        int epochs = 100;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (double[] instance : trainingData) {
                // Forward pass (calculate prediction without using predict method)
                double prediction = bias;
                for (int i = 0; i < weights.length; i++) {
                    prediction += weights[i] * instance[i];
                }
                prediction = sigmoid(prediction);
                
                // Simulate target (in real scenario, this would come from data)
                double target = instance[0] > 0.5 ? 1.0 : 0.0;
                
                // Calculate loss
                double loss = Math.pow(prediction - target, 2);
                totalLoss += loss;
                
                // Backward pass (simplified)
                double error = prediction - target;
                
                // Update weights
                for (int i = 0; i < weights.length; i++) {
                    weights[i] -= learningRate * error * instance[i];
                }
                
                // Update bias
                bias -= learningRate * error;
            }
            
            // Early stopping if loss is low enough
            if (totalLoss / trainingData.length < 0.01) {
                break;
            }
        }
        
        this.isTrained = true;
        System.out.println("Model training completed. Model is now trained.");
    }
    
    /**
     * Make a prediction for a single instance
     * 
     * @param instance input features
     * @return predicted value
     */
    public double predict(double[] instance) {
        if (!isTrained) {
            System.err.println("Predict called but model is not trained!");
            throw new IllegalStateException("Model must be trained before making predictions");
        }
        
        if (instance == null || instance.length != weights.length) {
            throw new IllegalArgumentException("Instance must have " + weights.length + " features");
        }
        
        // Linear combination
        double prediction = bias;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * instance[i];
        }
        
        // Apply sigmoid activation for classification-like output
        return sigmoid(prediction);
    }
    
    /**
     * Make predictions for multiple instances
     * 
     * @param instances array of input features
     * @return array of predictions
     */
    public double[] predict(double[][] instances) {
        if (instances == null || instances.length == 0) {
            throw new IllegalArgumentException("Instances cannot be null or empty");
        }
        
        double[] predictions = new double[instances.length];
        for (int i = 0; i < instances.length; i++) {
            predictions[i] = predict(instances[i]);
        }
        
        return predictions;
    }
    
    /**
     * Get feature importance scores
     * 
     * @return array of feature importance scores
     */
    public double[] getFeatureImportance() {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before getting feature importance");
        }
        
        // For linear models, feature importance is proportional to absolute weight values
        double[] importance = new double[weights.length];
        double maxWeight = 0.0;
        
        for (int i = 0; i < weights.length; i++) {
            importance[i] = Math.abs(weights[i]);
            maxWeight = Math.max(maxWeight, importance[i]);
        }
        
        // Normalize to [0, 1]
        if (maxWeight > 0) {
            for (int i = 0; i < importance.length; i++) {
                importance[i] /= maxWeight;
            }
        }
        
        return importance;
    }
    
    /**
     * Get model parameters
     * 
     * @return array containing weights and bias
     */
    public double[] getParameters() {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before getting parameters");
        }
        
        double[] parameters = new double[weights.length + 1];
        System.arraycopy(weights, 0, parameters, 0, weights.length);
        parameters[weights.length] = bias;
        
        return parameters;
    }
    
    /**
     * Set model parameters
     * 
     * @param parameters array containing weights and bias
     */
    public void setParameters(double[] parameters) {
        if (parameters == null || parameters.length < 1) {
            throw new IllegalArgumentException("Parameters cannot be null or empty");
        }
        
        this.weights = new double[parameters.length - 1];
        System.arraycopy(parameters, 0, this.weights, 0, weights.length);
        this.bias = parameters[parameters.length - 1];
        this.isTrained = true;
    }
    
    /**
     * Check if the model is trained
     * 
     * @return true if the model is trained
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Get the number of features the model expects
     * 
     * @return number of features
     */
    public int getNumFeatures() {
        return weights != null ? weights.length : 0;
    }
    
    /**
     * Sigmoid activation function
     * 
     * @param x input value
     * @return sigmoid output
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Get model summary information
     * 
     * @return string containing model information
     */
    public String getModelSummary() {
        if (!isTrained) {
            return "Model not trained";
        }
        
        StringBuilder summary = new StringBuilder();
        summary.append("Model Summary:\n");
        summary.append("Number of features: ").append(weights.length).append("\n");
        summary.append("Bias: ").append(String.format("%.4f", bias)).append("\n");
        summary.append("Weights: [");
        
        for (int i = 0; i < weights.length; i++) {
            summary.append(String.format("%.4f", weights[i]));
            if (i < weights.length - 1) {
                summary.append(", ");
            }
        }
        summary.append("]\n");
        
        return summary.toString();
    }
    
    /**
     * Calculate model complexity (number of parameters)
     * 
     * @return number of parameters
     */
    public int getModelComplexity() {
        return isTrained ? weights.length + 1 : 0;
    }
    
    /**
     * Reset the model to untrained state
     */
    public void reset() {
        this.weights = null;
        this.bias = 0.0;
        this.isTrained = false;
    }
}

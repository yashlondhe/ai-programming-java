package com.aiprogramming.ch17.utils;

import com.aiprogramming.utils.MatrixUtils;
import com.aiprogramming.utils.StatisticsUtils;
import com.aiprogramming.utils.ValidationUtils;
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
        ValidationUtils.validateMatrix(trainingData, "trainingData");
        
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
                ValidationUtils.validateVector(instance, "instance");
                
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
            throw new IllegalStateException("Model must be trained before making predictions");
        }
        
        ValidationUtils.validateVector(instance, "instance");
        if (instance.length != weights.length) {
            throw new IllegalArgumentException("Instance must have " + weights.length + " features");
        }
        
        double prediction = bias;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * instance[i];
        }
        
        return sigmoid(prediction);
    }
    
    /**
     * Make predictions for multiple instances
     * 
     * @param instances input features matrix
     * @return array of predictions
     */
    public double[] predictBatch(double[][] instances) {
        ValidationUtils.validateMatrix(instances, "instances");
        
        double[] predictions = new double[instances.length];
        for (int i = 0; i < instances.length; i++) {
            predictions[i] = predict(instances[i]);
        }
        
        return predictions;
    }
    
    /**
     * Get feature importance scores
     * 
     * @return array of importance scores
     */
    public double[] getFeatureImportance() {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before getting feature importance");
        }
        
        // Simple feature importance based on absolute weight values
        double[] importance = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            importance[i] = Math.abs(weights[i]);
        }
        
        // Normalize importance scores to sum to 1
        double sum = 0.0;
        for (double value : importance) {
            sum += value;
        }
        if (sum > 0) {
            for (int i = 0; i < importance.length; i++) {
                importance[i] /= sum;
            }
        }
        
        return importance;
    }
    
    /**
     * Get model performance metrics
     * 
     * @param testData test dataset
     * @param testLabels true labels
     * @return array with [accuracy, mse]
     */
    public double[] getPerformance(double[][] testData, double[] testLabels) {
        ValidationUtils.validateMatrix(testData, "testData");
        ValidationUtils.validateVector(testLabels, "testLabels");
        if (testData.length != testLabels.length) {
            throw new IllegalArgumentException("Test data and test labels must have the same length");
        }
        
        double[] predictions = predictBatch(testData);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if ((predictions[i] > 0.5 && testLabels[i] > 0.5) || 
                (predictions[i] <= 0.5 && testLabels[i] <= 0.5)) {
                correct++;
            }
        }
        double accuracy = (double) correct / predictions.length;
        
        // Calculate mean squared error
        double mse = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            mse += Math.pow(predictions[i] - testLabels[i], 2);
        }
        mse /= predictions.length;
        
        return new double[]{accuracy, mse};
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

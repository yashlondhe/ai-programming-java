package com.aiprogramming.ch18;

import java.util.Map;

/**
 * Interface for machine learning models
 */
public interface MLModel {
    
    /**
     * Train the model on the given data
     */
    void train(double[][] features, double[] targets);
    
    /**
     * Make predictions on new data
     */
    double[] predict(double[][] features);
    
    /**
     * Evaluate the model on test data
     */
    double evaluate(double[][] features, double[] targets);
    
    /**
     * Set hyperparameters for the model
     */
    void setHyperparameters(Map<String, Object> hyperparameters);
    
    /**
     * Get the current hyperparameters
     */
    Map<String, Object> getHyperparameters();
}

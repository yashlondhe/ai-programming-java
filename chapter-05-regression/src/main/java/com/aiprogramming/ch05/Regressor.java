package com.aiprogramming.ch05;

import java.util.List;
import java.util.Map;

/**
 * Base interface for all regression algorithms
 */
public interface Regressor {
    /**
     * Trains the regressor on the provided training data
     */
    void train(List<RegressionDataPoint> trainingData);
    
    /**
     * Predicts the target value for a given set of features
     */
    double predict(Map<String, Double> features);
    
    /**
     * Gets the name of the regression algorithm
     */
    default String getName() {
        return this.getClass().getSimpleName();
    }
}
package com.aiprogramming.ch04;

import java.util.List;
import java.util.Map;

/**
 * Base interface for all classification algorithms.
 * Defines the common contract for training and prediction.
 */
public interface Classifier {
    
    /**
     * Trains the classifier on the provided training data.
     * 
     * @param trainingData List of training data points
     */
    void train(List<ClassificationDataPoint> trainingData);
    
    /**
     * Predicts the class label for a given set of features.
     * 
     * @param features Map of feature names to feature values
     * @return Predicted class label
     */
    String predict(Map<String, Double> features);
    
    /**
     * Gets the name of the classifier algorithm.
     * 
     * @return Algorithm name
     */
    default String getName() {
        return this.getClass().getSimpleName();
    }
}

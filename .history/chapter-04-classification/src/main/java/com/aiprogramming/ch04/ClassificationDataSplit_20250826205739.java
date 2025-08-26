package com.aiprogramming.ch04;

import java.util.List;

/**
 * Holds training and test data for classification tasks.
 */
public class ClassificationDataSplit {
    
    private final List<ClassificationDataPoint> trainingData;
    private final List<ClassificationDataPoint> testData;
    
    public ClassificationDataSplit(List<ClassificationDataPoint> trainingData, 
                                 List<ClassificationDataPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }
    
    /**
     * Gets the training data
     */
    public List<ClassificationDataPoint> getTrainingData() {
        return trainingData;
    }
    
    /**
     * Gets the test data
     */
    public List<ClassificationDataPoint> getTestData() {
        return testData;
    }
    
    /**
     * Gets the size of training data
     */
    public int getTrainingSize() {
        return trainingData.size();
    }
    
    /**
     * Gets the size of test data
     */
    public int getTestSize() {
        return testData.size();
    }
    
    /**
     * Gets the total size of the dataset
     */
    public int getTotalSize() {
        return trainingData.size() + testData.size();
    }
}

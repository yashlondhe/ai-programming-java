package com.aiprogramming.ch05;

import java.util.*;

/**
 * Container class for train/test split results
 */
public class RegressionDataSplit {
    private final List<RegressionDataPoint> trainingData;
    private final List<RegressionDataPoint> testData;
    
    public RegressionDataSplit(List<RegressionDataPoint> trainingData, List<RegressionDataPoint> testData) {
        this.trainingData = new ArrayList<>(trainingData);
        this.testData = new ArrayList<>(testData);
    }
    
    public List<RegressionDataPoint> getTrainingData() {
        return new ArrayList<>(trainingData);
    }
    
    public List<RegressionDataPoint> getTestData() {
        return new ArrayList<>(testData);
    }
}
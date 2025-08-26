package com.aiprogramming.ch05;

import java.util.*;

/**
 * Utility class for splitting regression data into training and testing sets
 */
public class RegressionDataSplitter {
    
    private final double trainRatio;
    private final Random random;
    
    public RegressionDataSplitter(double trainRatio) {
        this(trainRatio, new Random());
    }
    
    public RegressionDataSplitter(double trainRatio, Random random) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        this.trainRatio = trainRatio;
        this.random = random;
    }
    
    public RegressionDataSplit split(List<RegressionDataPoint> data) {
        List<RegressionDataPoint> shuffledData = new ArrayList<>(data);
        Collections.shuffle(shuffledData, random);
        
        int trainSize = (int) (data.size() * trainRatio);
        
        List<RegressionDataPoint> trainData = shuffledData.subList(0, trainSize);
        List<RegressionDataPoint> testData = shuffledData.subList(trainSize, shuffledData.size());
        
        return new RegressionDataSplit(trainData, testData);
    }
}
package com.aiprogramming.ch18;

/**
 * Represents a split of data into training and test sets
 */
public class DataSplit {
    
    private final double[][] trainFeatures;
    private final double[] trainTargets;
    private final double[][] testFeatures;
    private final double[] testTargets;
    
    public DataSplit(double[][] trainFeatures, double[] trainTargets, 
                    double[][] testFeatures, double[] testTargets) {
        this.trainFeatures = trainFeatures;
        this.trainTargets = trainTargets;
        this.testFeatures = testFeatures;
        this.testTargets = testTargets;
    }
    
    public double[][] getTrainFeatures() {
        return trainFeatures;
    }
    
    public double[] getTrainTargets() {
        return trainTargets;
    }
    
    public double[][] getTestFeatures() {
        return testFeatures;
    }
    
    public double[] getTestTargets() {
        return testTargets;
    }
}

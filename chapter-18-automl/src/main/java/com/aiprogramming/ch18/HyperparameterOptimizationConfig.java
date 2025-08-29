package com.aiprogramming.ch18;

/**
 * Configuration for hyperparameter optimization
 */
public class HyperparameterOptimizationConfig {
    
    public enum OptimizationMethod {
        BAYESIAN,
        GRID_SEARCH,
        RANDOM_SEARCH
    }
    
    private final OptimizationMethod optimizationMethod;
    private final int maxTrials;
    private final int cvFolds;
    private final long randomSeed;
    private final double tolerance;
    
    public HyperparameterOptimizationConfig() {
        this(OptimizationMethod.RANDOM_SEARCH, 50, 5, 42L, 1e-6);
    }
    
    public HyperparameterOptimizationConfig(OptimizationMethod optimizationMethod, 
                                          int maxTrials, int cvFolds, long randomSeed, double tolerance) {
        this.optimizationMethod = optimizationMethod;
        this.maxTrials = maxTrials;
        this.cvFolds = cvFolds;
        this.randomSeed = randomSeed;
        this.tolerance = tolerance;
    }
    
    public OptimizationMethod getOptimizationMethod() {
        return optimizationMethod;
    }
    
    public int getMaxTrials() {
        return maxTrials;
    }
    
    public int getCvFolds() {
        return cvFolds;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    public double getTolerance() {
        return tolerance;
    }
}

package com.aiprogramming.ch18;

/**
 * Configuration for Neural Architecture Search
 */
public class NASConfig {
    
    private final int maxTrials;
    private final int minLayers;
    private final int maxLayers;
    private final int minLayerSize;
    private final int maxLayerSize;
    private final int inputSize;
    private final int outputSize;
    private final double maxDropout;
    private final String[] activationFunctions;
    private final long randomSeed;
    
    public NASConfig() {
        this(50, 1, 5, 10, 100, 10, 1, 0.5, 
             new String[]{"relu", "tanh", "sigmoid"}, 42L);
    }
    
    public NASConfig(int maxTrials, int minLayers, int maxLayers, int minLayerSize, 
                    int maxLayerSize, int inputSize, int outputSize, double maxDropout,
                    String[] activationFunctions, long randomSeed) {
        this.maxTrials = maxTrials;
        this.minLayers = minLayers;
        this.maxLayers = maxLayers;
        this.minLayerSize = minLayerSize;
        this.maxLayerSize = maxLayerSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.maxDropout = maxDropout;
        this.activationFunctions = activationFunctions;
        this.randomSeed = randomSeed;
    }
    
    public int getMaxTrials() {
        return maxTrials;
    }
    
    public int getMinLayers() {
        return minLayers;
    }
    
    public int getMaxLayers() {
        return maxLayers;
    }
    
    public int getMinLayerSize() {
        return minLayerSize;
    }
    
    public int getMaxLayerSize() {
        return maxLayerSize;
    }
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    public double getMaxDropout() {
        return maxDropout;
    }
    
    public String[] getActivationFunctions() {
        return activationFunctions;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
}

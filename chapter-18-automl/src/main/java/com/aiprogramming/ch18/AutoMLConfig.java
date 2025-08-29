package com.aiprogramming.ch18;

/**
 * Configuration class for AutoML pipeline
 */
public class AutoMLConfig {
    
    private final HyperparameterOptimizationConfig optimizationConfig;
    private final FeatureSelectionConfig featureSelectionConfig;
    private final ModelSelectionConfig modelSelectionConfig;
    private final int maxTimeSeconds;
    private final int maxTrials;
    private final int cvFolds;
    
    public AutoMLConfig(Builder builder) {
        this.optimizationConfig = builder.optimizationConfig;
        this.featureSelectionConfig = builder.featureSelectionConfig;
        this.modelSelectionConfig = builder.modelSelectionConfig;
        this.maxTimeSeconds = builder.maxTimeSeconds;
        this.maxTrials = builder.maxTrials;
        this.cvFolds = builder.cvFolds;
    }
    
    public HyperparameterOptimizationConfig getOptimizationConfig() {
        return optimizationConfig;
    }
    
    public FeatureSelectionConfig getFeatureSelectionConfig() {
        return featureSelectionConfig;
    }
    
    public ModelSelectionConfig getModelSelectionConfig() {
        return modelSelectionConfig;
    }
    
    public int getMaxTimeSeconds() {
        return maxTimeSeconds;
    }
    
    public int getMaxTrials() {
        return maxTrials;
    }
    
    public int getCvFolds() {
        return cvFolds;
    }
    
    public static class Builder {
        private HyperparameterOptimizationConfig optimizationConfig = new HyperparameterOptimizationConfig();
        private FeatureSelectionConfig featureSelectionConfig = new FeatureSelectionConfig();
        private ModelSelectionConfig modelSelectionConfig = new ModelSelectionConfig();
        private int maxTimeSeconds = 3600; // 1 hour default
        private int maxTrials = 100;
        private int cvFolds = 5;
        
        public Builder optimizationConfig(HyperparameterOptimizationConfig config) {
            this.optimizationConfig = config;
            return this;
        }
        
        public Builder featureSelectionConfig(FeatureSelectionConfig config) {
            this.featureSelectionConfig = config;
            return this;
        }
        
        public Builder modelSelectionConfig(ModelSelectionConfig config) {
            this.modelSelectionConfig = config;
            return this;
        }
        
        public Builder maxTimeSeconds(int maxTimeSeconds) {
            this.maxTimeSeconds = maxTimeSeconds;
            return this;
        }
        
        public Builder maxTrials(int maxTrials) {
            this.maxTrials = maxTrials;
            return this;
        }
        
        public Builder cvFolds(int cvFolds) {
            this.cvFolds = cvFolds;
            return this;
        }
        
        public AutoMLConfig build() {
            return new AutoMLConfig(this);
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
}

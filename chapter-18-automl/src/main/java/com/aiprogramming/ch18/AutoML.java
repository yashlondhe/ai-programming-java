package com.aiprogramming.ch18;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

/**
 * Automated Machine Learning (AutoML) framework
 * Provides automated hyperparameter tuning, feature selection, and model selection
 */
public class AutoML {
    
    private final HyperparameterOptimizer hyperparameterOptimizer;
    private final FeatureSelector featureSelector;
    private final ModelSelector modelSelector;
    private final AutoMLConfig config;
    
    public AutoML(AutoMLConfig config) {
        this.config = config;
        this.hyperparameterOptimizer = new HyperparameterOptimizer(config.getOptimizationConfig());
        this.featureSelector = new FeatureSelector(config.getFeatureSelectionConfig());
        this.modelSelector = new ModelSelector(config.getModelSelectionConfig());
    }
    
    /**
     * Run complete AutoML pipeline
     */
    public AutoMLResult runAutoML(double[][] features, double[] targets) {
        System.out.println("Starting AutoML pipeline...");
        
        // Step 1: Feature Selection
        System.out.println("Step 1: Feature Selection");
        FeatureSelectionResult featureResult = featureSelector.selectFeatures(features, targets);
        
        // Step 2: Model Selection
        System.out.println("Step 2: Model Selection");
        ModelSelectionResult modelResult = modelSelector.selectBestModel(
            featureResult.getSelectedFeatures(), targets);
        
        // Step 3: Hyperparameter Optimization
        System.out.println("Step 3: Hyperparameter Optimization");
        HyperparameterOptimizationResult hpResult = hyperparameterOptimizer.optimize(
            modelResult.getBestModel(), 
            featureResult.getSelectedFeatures(), 
            targets);
        
        return new AutoMLResult(featureResult, modelResult, hpResult);
    }
    
    /**
     * Run hyperparameter optimization only
     */
    public HyperparameterOptimizationResult optimizeHyperparameters(
            MLModel model, double[][] features, double[] targets) {
        return hyperparameterOptimizer.optimize(model, features, targets);
    }
    
    /**
     * Run feature selection only
     */
    public FeatureSelectionResult selectFeatures(double[][] features, double[] targets) {
        return featureSelector.selectFeatures(features, targets);
    }
    
    /**
     * Run model selection only
     */
    public ModelSelectionResult selectBestModel(double[][] features, double[] targets) {
        return modelSelector.selectBestModel(features, targets);
    }
    
    /**
     * Get the configuration
     */
    public AutoMLConfig getConfig() {
        return config;
    }
}

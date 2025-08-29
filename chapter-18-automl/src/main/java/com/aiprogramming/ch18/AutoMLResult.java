package com.aiprogramming.ch18;

/**
 * Result of AutoML pipeline execution
 */
public class AutoMLResult {
    
    private final FeatureSelectionResult featureSelectionResult;
    private final ModelSelectionResult modelSelectionResult;
    private final HyperparameterOptimizationResult hyperparameterOptimizationResult;
    
    public AutoMLResult(FeatureSelectionResult featureSelectionResult,
                       ModelSelectionResult modelSelectionResult,
                       HyperparameterOptimizationResult hyperparameterOptimizationResult) {
        this.featureSelectionResult = featureSelectionResult;
        this.modelSelectionResult = modelSelectionResult;
        this.hyperparameterOptimizationResult = hyperparameterOptimizationResult;
    }
    
    public FeatureSelectionResult getFeatureSelectionResult() {
        return featureSelectionResult;
    }
    
    public ModelSelectionResult getModelSelectionResult() {
        return modelSelectionResult;
    }
    
    public HyperparameterOptimizationResult getHyperparameterOptimizationResult() {
        return hyperparameterOptimizationResult;
    }
    
    /**
     * Get the final optimized model
     */
    public MLModel getFinalModel() {
        return modelSelectionResult.getBestModel();
    }
    
    /**
     * Get the best hyperparameters
     */
    public java.util.Map<String, Object> getBestHyperparameters() {
        return hyperparameterOptimizationResult.getBestParameters();
    }
    
    /**
     * Get the best score achieved
     */
    public double getBestScore() {
        return hyperparameterOptimizationResult.getBestScore();
    }
    
    /**
     * Print summary of AutoML results
     */
    public void printSummary() {
        System.out.println("\n=== AutoML Pipeline Results ===");
        System.out.println("Best Model: " + modelSelectionResult.getBestModel().getClass().getSimpleName());
        System.out.println("Best Score: " + String.format("%.4f", getBestScore()));
        System.out.println("Selected Features: " + featureSelectionResult.getSelectedFeatureIndices().length);
        System.out.println("Best Hyperparameters: " + getBestHyperparameters());
        
        System.out.println("\nFeature Selection Results:");
        featureSelectionResult.printSummary();
        
        System.out.println("\nModel Selection Results:");
        modelSelectionResult.printSummary();
        
        System.out.println("\nHyperparameter Optimization Results:");
        hyperparameterOptimizationResult.printSummary();
    }
}

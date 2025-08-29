package com.aiprogramming.ch18;

import java.util.*;

/**
 * Simplified AutoML demo focusing on working components
 */
public class SimpleAutoMLDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Simple AutoML Demo ===\n");
        
        // Generate smaller, simpler dataset
        double[][] features = generateSimpleData(500, 10);
        double[] targets = generateSimpleTargets(features);
        
        System.out.println("Generated dataset with " + features.length + " samples and " + 
                          features[0].length + " features");
        
        // Demonstrate individual components
        demonstrateFeatureSelection(features, targets);
        demonstrateModelSelection(features, targets);
        demonstrateHyperparameterOptimization(features, targets);
        demonstrateSimpleAutoML(features, targets);
        
        System.out.println("\n=== Demo Complete ===");
    }
    
    /**
     * Demonstrate feature selection
     */
    private static void demonstrateFeatureSelection(double[][] features, double[] targets) {
        System.out.println("\n=== Feature Selection Demo ===");
        
        // Test correlation-based feature selection
        System.out.println("Testing correlation-based feature selection:");
        FeatureSelectionConfig config = new FeatureSelectionConfig(
            FeatureSelectionConfig.SelectionMethod.CORRELATION, 5, 0.01, true);
        FeatureSelector selector = new FeatureSelector(config);
        FeatureSelectionResult result = selector.selectFeatures(features, targets);
        result.printSummary();
    }
    
    /**
     * Demonstrate model selection
     */
    private static void demonstrateModelSelection(double[][] features, double[] targets) {
        System.out.println("\n=== Model Selection Demo ===");
        
        // Configure model selection with only working models
        Set<Class<? extends MLModel>> models = new HashSet<>();
        models.add(LinearRegression.class);
        models.add(LogisticRegression.class);
        
        ModelSelectionConfig config = new ModelSelectionConfig(models, 
            ModelSelectionConfig.EvaluationMetric.MEAN_SQUARED_ERROR, true, 3);
        ModelSelector selector = new ModelSelector(config);
        
        ModelSelectionResult result = selector.selectBestModel(features, targets);
        result.printSummary();
    }
    
    /**
     * Demonstrate hyperparameter optimization
     */
    private static void demonstrateHyperparameterOptimization(double[][] features, double[] targets) {
        System.out.println("\n=== Hyperparameter Optimization Demo ===");
        
        // Test random search optimization
        HyperparameterOptimizationConfig config = new HyperparameterOptimizationConfig(
            HyperparameterOptimizationConfig.OptimizationMethod.RANDOM_SEARCH, 10, 3, 42L, 1e-6);
        
        LinearRegression model = new LinearRegression();
        HyperparameterOptimizer optimizer = new HyperparameterOptimizer(config);
        HyperparameterOptimizationResult result = optimizer.optimize(model, features, targets);
        result.printSummary();
    }
    
    /**
     * Demonstrate simple AutoML pipeline
     */
    private static void demonstrateSimpleAutoML(double[][] features, double[] targets) {
        System.out.println("\n=== Simple AutoML Pipeline ===");
        
        // Configure AutoML with only working components
        AutoMLConfig config = AutoMLConfig.builder()
            .maxTrials(10)
            .cvFolds(3)
            .build();
        
        AutoML autoML = new AutoML(config);
        
        // Run AutoML pipeline
        AutoMLResult result = autoML.runAutoML(features, targets);
        result.printSummary();
    }
    
    /**
     * Generate simple features
     */
    private static double[][] generateSimpleData(int numSamples, int numFeatures) {
        Random random = new Random(42);
        double[][] features = new double[numSamples][numFeatures];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                features[i][j] = random.nextGaussian();
            }
        }
        
        return features;
    }
    
    /**
     * Generate simple target values
     */
    private static double[] generateSimpleTargets(double[][] features) {
        double[] targets = new double[features.length];
        Random random = new Random(42);
        
        for (int i = 0; i < features.length; i++) {
            // Create a simple linear relationship
            double prediction = 0.0;
            for (int j = 0; j < features[i].length; j++) {
                prediction += features[i][j] * 0.1;
            }
            
            // Add small noise
            targets[i] = prediction + random.nextGaussian() * 0.05;
        }
        
        return targets;
    }
}

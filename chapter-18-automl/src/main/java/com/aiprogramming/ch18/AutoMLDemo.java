package com.aiprogramming.ch18;

import java.util.*;
import com.aiprogramming.utils.MatrixUtils;
import com.aiprogramming.utils.StatisticsUtils;
import com.aiprogramming.utils.ValidationUtils;

/**
 * Comprehensive demonstration of AutoML and Neural Architecture Search
 */
public class AutoMLDemo {
    
    public static void main(String[] args) {
        System.out.println("=== AutoML and Neural Architecture Search Demo ===\n");
        
        // Generate sample data
        double[][] features = generateSampleData(1000, 20);
        double[] targets = generateTargets(features);
        
        System.out.println("Generated dataset with " + features.length + " samples and " + 
                          features[0].length + " features");
        
        // Demonstrate AutoML pipeline
        demonstrateAutoML(features, targets);
        
        // Demonstrate Neural Architecture Search
        demonstrateNAS(features, targets);
        
        // Demonstrate individual components
        demonstrateFeatureSelection(features, targets);
        demonstrateModelSelection(features, targets);
        demonstrateHyperparameterOptimization(features, targets);
        
        System.out.println("\n=== Demo Complete ===");
    }
    
    /**
     * Demonstrate complete AutoML pipeline
     */
    private static void demonstrateAutoML(double[][] features, double[] targets) {
        System.out.println("\n=== Complete AutoML Pipeline ===");
        
        // Configure AutoML
        AutoMLConfig config = AutoMLConfig.builder()
            .maxTrials(20)
            .cvFolds(3)
            .build();
        
        AutoML autoML = new AutoML(config);
        
        // Run complete pipeline
        AutoMLResult result = autoML.runAutoML(features, targets);
        
        // Print results
        result.printSummary();
    }
    
    /**
     * Demonstrate Neural Architecture Search
     */
    private static void demonstrateNAS(double[][] features, double[] targets) {
        System.out.println("\n=== Neural Architecture Search ===");
        
        // Configure NAS
        NASConfig nasConfig = new NASConfig(10, 1, 3, 5, 50, features[0].length, 1, 0.3,
                                           new String[]{"relu", "tanh"}, 42L);
        
        NeuralArchitectureSearch nas = new NeuralArchitectureSearch(nasConfig);
        
        // Run NAS
        NeuralArchitectureSearch.NASResult result = nas.search(features, targets);
        
        // Print results
        result.printSummary();
    }
    
    /**
     * Demonstrate feature selection
     */
    private static void demonstrateFeatureSelection(double[][] features, double[] targets) {
        System.out.println("\n=== Feature Selection Demo ===");
        
        // Test different feature selection methods
        FeatureSelectionConfig[] configs = {
            new FeatureSelectionConfig(FeatureSelectionConfig.SelectionMethod.CORRELATION, 5, 0.01, true),
            new FeatureSelectionConfig(FeatureSelectionConfig.SelectionMethod.MUTUAL_INFORMATION, 5, 0.01, true),
            new FeatureSelectionConfig(FeatureSelectionConfig.SelectionMethod.RANDOM_FOREST_IMPORTANCE, 5, 0.01, true)
        };
        
        for (FeatureSelectionConfig config : configs) {
            System.out.println("\nTesting " + config.getSelectionMethod() + " feature selection:");
            FeatureSelector selector = new FeatureSelector(config);
            FeatureSelectionResult result = selector.selectFeatures(features, targets);
            result.printSummary();
        }
    }
    
    /**
     * Demonstrate model selection
     */
    private static void demonstrateModelSelection(double[][] features, double[] targets) {
        System.out.println("\n=== Model Selection Demo ===");
        
        // Configure model selection
        ModelSelectionConfig config = new ModelSelectionConfig();
        ModelSelector selector = new ModelSelector(config);
        
        // Run model selection
        ModelSelectionResult result = selector.selectBestModel(features, targets);
        
        // Print results
        result.printSummary();
    }
    
    /**
     * Demonstrate hyperparameter optimization
     */
    private static void demonstrateHyperparameterOptimization(double[][] features, double[] targets) {
        System.out.println("\n=== Hyperparameter Optimization Demo ===");
        
        // Test different optimization methods
        HyperparameterOptimizationConfig[] configs = {
            new HyperparameterOptimizationConfig(HyperparameterOptimizationConfig.OptimizationMethod.RANDOM_SEARCH, 10, 3, 42L, 1e-6),
            new HyperparameterOptimizationConfig(HyperparameterOptimizationConfig.OptimizationMethod.GRID_SEARCH, 10, 3, 42L, 1e-6),
            new HyperparameterOptimizationConfig(HyperparameterOptimizationConfig.OptimizationMethod.BAYESIAN, 10, 3, 42L, 1e-6)
        };
        
        LinearRegression model = new LinearRegression();
        
        for (HyperparameterOptimizationConfig config : configs) {
            System.out.println("\nTesting " + config.getOptimizationMethod() + " optimization:");
            HyperparameterOptimizer optimizer = new HyperparameterOptimizer(config);
            HyperparameterOptimizationResult result = optimizer.optimize(model, features, targets);
            result.printSummary();
        }
    }
    
    /**
     * Generate sample features
     */
    private static double[][] generateSampleData(int numSamples, int numFeatures) {
        return MatrixUtils.randomNormal(numSamples, numFeatures, 0.0, 1.0, 42L);
    }
    
    /**
     * Generate target values based on features
     */
    private static double[] generateTargets(double[][] features) {
        double[] targets = new double[features.length];
        Random random = new Random(42);
        
        for (int i = 0; i < features.length; i++) {
            // Create a simple linear relationship with some noise
            double prediction = 0.0;
            for (int j = 0; j < features[i].length; j++) {
                prediction += features[i][j] * (j + 1) * 0.1;
            }
            
            // Add noise
            targets[i] = prediction + random.nextGaussian() * 0.1;
        }
        
        return targets;
    }
    
    /**
     * Demonstrate advanced AutoML features
     */
    public static void demonstrateAdvancedFeatures() {
        System.out.println("\n=== Advanced AutoML Features ===");
        
        // Custom AutoML configuration
        AutoMLConfig customConfig = AutoMLConfig.builder()
            .maxTrials(50)
            .maxTimeSeconds(300)
            .cvFolds(5)
            .build();
        
        System.out.println("Custom AutoML configuration:");
        System.out.println("Max trials: " + customConfig.getMaxTrials());
        System.out.println("Max time: " + customConfig.getMaxTimeSeconds() + " seconds");
        System.out.println("CV folds: " + customConfig.getCvFolds());
        
        // Custom NAS configuration
        NASConfig customNASConfig = new NASConfig(20, 2, 4, 20, 80, 15, 1, 0.4,
                                                 new String[]{"relu", "tanh", "sigmoid"}, 123L);
        
        System.out.println("\nCustom NAS configuration:");
        System.out.println("Max trials: " + customNASConfig.getMaxTrials());
        System.out.println("Layer range: " + customNASConfig.getMinLayers() + "-" + customNASConfig.getMaxLayers());
        System.out.println("Layer size range: " + customNASConfig.getMinLayerSize() + "-" + customNASConfig.getMaxLayerSize());
    }
    
    /**
     * Demonstrate performance comparison
     */
    public static void demonstratePerformanceComparison(double[][] features, double[] targets) {
        System.out.println("\n=== Performance Comparison ===");
        
        // Compare different AutoML configurations
        AutoMLConfig[] configs = {
            AutoMLConfig.builder().maxTrials(10).build(),
            AutoMLConfig.builder().maxTrials(20).build(),
            AutoMLConfig.builder().maxTrials(30).build()
        };
        
        for (AutoMLConfig config : configs) {
            System.out.println("\nTesting with " + config.getMaxTrials() + " trials:");
            
            long startTime = System.currentTimeMillis();
            AutoML autoML = new AutoML(config);
            AutoMLResult result = autoML.runAutoML(features, targets);
            long endTime = System.currentTimeMillis();
            
            System.out.println("Time taken: " + (endTime - startTime) + " ms");
            System.out.println("Best score: " + String.format("%.4f", result.getBestScore()));
        }
    }
}

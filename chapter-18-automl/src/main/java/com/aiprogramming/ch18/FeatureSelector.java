package com.aiprogramming.ch18;

import java.util.*;

/**
 * Feature selection using various methods
 */
public class FeatureSelector {
    
    private final FeatureSelectionConfig config;
    
    public FeatureSelector(FeatureSelectionConfig config) {
        this.config = config;
    }
    
    /**
     * Select features using the configured method
     */
    public FeatureSelectionResult selectFeatures(double[][] features, double[] targets) {
        System.out.println("Starting feature selection using " + config.getSelectionMethod());
        
        switch (config.getSelectionMethod()) {
            case CORRELATION:
                return selectByCorrelation(features, targets);
            case MUTUAL_INFORMATION:
                return selectByMutualInformation(features, targets);
            case RECURSIVE_FEATURE_ELIMINATION:
                return selectByRecursiveFeatureElimination(features, targets);
            case LASSO:
                return selectByLasso(features, targets);
            case RANDOM_FOREST_IMPORTANCE:
                return selectByRandomForestImportance(features, targets);
            default:
                return selectByCorrelation(features, targets);
        }
    }
    
    /**
     * Select features based on correlation with target
     */
    private FeatureSelectionResult selectByCorrelation(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        double[] correlations = new double[numFeatures];
        
        // Calculate correlation for each feature
        for (int i = 0; i < numFeatures; i++) {
            double[] featureValues = new double[features.length];
            for (int j = 0; j < features.length; j++) {
                featureValues[j] = features[j][i];
            }
            correlations[i] = Math.abs(calculateCorrelation(featureValues, targets));
        }
        
        // Select top features
        return selectTopFeatures(correlations, features);
    }
    
    /**
     * Select features based on mutual information
     */
    private FeatureSelectionResult selectByMutualInformation(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        double[] mutualInfo = new double[numFeatures];
        
        // Calculate mutual information for each feature
        for (int i = 0; i < numFeatures; i++) {
            double[] featureValues = new double[features.length];
            for (int j = 0; j < features.length; j++) {
                featureValues[j] = features[j][i];
            }
            mutualInfo[i] = calculateMutualInformation(featureValues, targets);
        }
        
        return selectTopFeatures(mutualInfo, features);
    }
    
    /**
     * Select features using recursive feature elimination
     */
    private FeatureSelectionResult selectByRecursiveFeatureElimination(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        List<Integer> remainingFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            remainingFeatures.add(i);
        }
        
        while (remainingFeatures.size() > config.getMaxFeatures()) {
            // Train model with remaining features
            double[][] currentFeatures = selectFeatures(features, remainingFeatures);
            LinearRegression model = new LinearRegression();
            model.train(currentFeatures, targets);
            
            // Find feature with lowest importance
            double[] coefficients = model.getCoefficients();
            int worstFeatureIndex = 0;
            double minImportance = Math.abs(coefficients[0]);
            
            for (int i = 1; i < coefficients.length; i++) {
                if (Math.abs(coefficients[i]) < minImportance) {
                    minImportance = Math.abs(coefficients[i]);
                    worstFeatureIndex = i;
                }
            }
            
            // Remove worst feature
            remainingFeatures.remove(worstFeatureIndex);
        }
        
        int[] selectedIndices = remainingFeatures.stream().mapToInt(Integer::intValue).toArray();
        double[][] selectedFeatures = selectFeatures(features, selectedIndices);
        
        return new FeatureSelectionResult(selectedIndices, selectedFeatures, 
                                        new double[selectedIndices.length]);
    }
    
    /**
     * Select features using Lasso regularization
     */
    private FeatureSelectionResult selectByLasso(double[][] features, double[] targets) {
        // Simple Lasso implementation
        double lambda = 0.1; // regularization parameter
        LinearRegression model = new LinearRegression();
        model.setHyperparameters(Map.of("regularization", lambda, "regularizationType", "lasso"));
        model.train(features, targets);
        
        double[] coefficients = model.getCoefficients();
        List<Integer> selectedIndices = new ArrayList<>();
        
        for (int i = 0; i < coefficients.length; i++) {
            if (Math.abs(coefficients[i]) > config.getThreshold()) {
                selectedIndices.add(i);
            }
        }
        
        int[] indices = selectedIndices.stream().mapToInt(Integer::intValue).toArray();
        double[][] selectedFeatures = selectFeatures(features, indices);
        
        return new FeatureSelectionResult(indices, selectedFeatures, coefficients);
    }
    
    /**
     * Select features using Random Forest importance
     */
    private FeatureSelectionResult selectByRandomForestImportance(double[][] features, double[] targets) {
        RandomForest rf = new RandomForest();
        rf.setHyperparameters(Map.of("numTrees", 10, "maxDepth", 5));
        rf.train(features, targets);
        
        double[] importance = rf.getFeatureImportance();
        return selectTopFeatures(importance, features);
    }
    
    /**
     * Select top features based on importance scores
     */
    private FeatureSelectionResult selectTopFeatures(double[] importance, double[][] features) {
        // Create feature index and importance pairs
        List<FeatureImportance> featureImportances = new ArrayList<>();
        for (int i = 0; i < importance.length; i++) {
            featureImportances.add(new FeatureImportance(i, importance[i]));
        }
        
        // Sort by importance (descending)
        featureImportances.sort((a, b) -> Double.compare(b.getImportance(), a.getImportance()));
        
        // Select top features
        int numToSelect = Math.min(config.getMaxFeatures(), featureImportances.size());
        int[] selectedIndices = new int[numToSelect];
        double[] selectedImportance = new double[numToSelect];
        
        for (int i = 0; i < numToSelect; i++) {
            selectedIndices[i] = featureImportances.get(i).getIndex();
            selectedImportance[i] = featureImportances.get(i).getImportance();
        }
        
        double[][] selectedFeatures = selectFeatures(features, selectedIndices);
        
        return new FeatureSelectionResult(selectedIndices, selectedFeatures, selectedImportance);
    }
    
    /**
     * Select specific features from the original feature matrix
     */
    private double[][] selectFeatures(double[][] features, List<Integer> indices) {
        double[][] selected = new double[features.length][indices.size()];
        for (int i = 0; i < features.length; i++) {
            for (int j = 0; j < indices.size(); j++) {
                selected[i][j] = features[i][indices.get(j)];
            }
        }
        return selected;
    }
    
    /**
     * Select specific features from the original feature matrix
     */
    private double[][] selectFeatures(double[][] features, int[] indices) {
        double[][] selected = new double[features.length][indices.length];
        for (int i = 0; i < features.length; i++) {
            for (int j = 0; j < indices.length; j++) {
                selected[i][j] = features[i][indices[j]];
            }
        }
        return selected;
    }
    
    /**
     * Calculate correlation between two arrays
     */
    private double calculateCorrelation(double[] x, double[] y) {
        double meanX = Arrays.stream(x).average().orElse(0.0);
        double meanY = Arrays.stream(y).average().orElse(0.0);
        
        double numerator = 0.0;
        double sumXSquared = 0.0;
        double sumYSquared = 0.0;
        
        for (int i = 0; i < x.length; i++) {
            double xDiff = x[i] - meanX;
            double yDiff = y[i] - meanY;
            numerator += xDiff * yDiff;
            sumXSquared += xDiff * xDiff;
            sumYSquared += yDiff * yDiff;
        }
        
        double denominator = Math.sqrt(sumXSquared * sumYSquared);
        return denominator == 0 ? 0 : numerator / denominator;
    }
    
    /**
     * Calculate mutual information (simplified version)
     */
    private double calculateMutualInformation(double[] x, double[] y) {
        // Simplified mutual information calculation
        // In practice, this would use histogram-based estimation
        return Math.abs(calculateCorrelation(x, y));
    }
    
    /**
     * Helper class for feature importance
     */
    private static class FeatureImportance {
        private final int index;
        private final double importance;
        
        public FeatureImportance(int index, double importance) {
            this.index = index;
            this.importance = importance;
        }
        
        public int getIndex() {
            return index;
        }
        
        public double getImportance() {
            return importance;
        }
    }
}

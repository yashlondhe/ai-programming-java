package com.aiprogramming.ch03;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Provides methods for feature selection in machine learning datasets.
 */
public class FeatureSelector {
    
    /**
     * Correlation-based feature selection
     */
    public List<String> selectByCorrelation(Dataset dataset, String targetFeature, double threshold) {
        List<String> selectedFeatures = new ArrayList<>();
        
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double correlation = calculateCorrelation(dataset, feature, targetFeature);
                if (Math.abs(correlation) > threshold) {
                    selectedFeatures.add(feature);
                }
            }
        }
        
        return selectedFeatures;
    }
    
    /**
     * Variance-based feature selection
     */
    public List<String> selectByVariance(Dataset dataset, double threshold) {
        return dataset.getFeatureNames().stream()
                .filter(feature -> calculateVariance(dataset, feature) > threshold)
                .collect(Collectors.toList());
    }
    
    /**
     * Mutual information-based selection (simplified)
     */
    public List<String> selectByMutualInformation(Dataset dataset, String targetFeature, int topK) {
        Map<String, Double> miScores = new HashMap<>();
        
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double mi = calculateMutualInformation(dataset, feature, targetFeature);
                miScores.put(feature, mi);
            }
        }
        
        return miScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
    
    /**
     * Forward selection algorithm
     */
    public List<String> forwardSelection(Dataset dataset, String targetFeature, int maxFeatures) {
        List<String> selectedFeatures = new ArrayList<>();
        List<String> availableFeatures = new ArrayList<>(dataset.getFeatureNames());
        availableFeatures.remove(targetFeature);
        
        for (int i = 0; i < maxFeatures && !availableFeatures.isEmpty(); i++) {
            String bestFeature = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (String feature : availableFeatures) {
                List<String> currentFeatures = new ArrayList<>(selectedFeatures);
                currentFeatures.add(feature);
                
                double score = evaluateFeatureSet(dataset, currentFeatures, targetFeature);
                if (score > bestScore) {
                    bestScore = score;
                    bestFeature = feature;
                }
            }
            
            if (bestFeature != null) {
                selectedFeatures.add(bestFeature);
                availableFeatures.remove(bestFeature);
            }
        }
        
        return selectedFeatures;
    }
    
    /**
     * Backward elimination algorithm
     */
    public List<String> backwardElimination(Dataset dataset, String targetFeature, int minFeatures) {
        List<String> selectedFeatures = new ArrayList<>(dataset.getFeatureNames());
        selectedFeatures.remove(targetFeature);
        
        while (selectedFeatures.size() > minFeatures) {
            String worstFeature = null;
            double worstScore = Double.POSITIVE_INFINITY;
            
            for (String feature : selectedFeatures) {
                List<String> currentFeatures = new ArrayList<>(selectedFeatures);
                currentFeatures.remove(feature);
                
                double score = evaluateFeatureSet(dataset, currentFeatures, targetFeature);
                if (score < worstScore) {
                    worstScore = score;
                    worstFeature = feature;
                }
            }
            
            if (worstFeature != null) {
                selectedFeatures.remove(worstFeature);
            } else {
                break;
            }
        }
        
        return selectedFeatures;
    }
    
    /**
     * Recursive feature elimination (simplified)
     */
    public List<String> recursiveFeatureElimination(Dataset dataset, String targetFeature, int nFeatures) {
        List<String> selectedFeatures = new ArrayList<>(dataset.getFeatureNames());
        selectedFeatures.remove(targetFeature);
        
        while (selectedFeatures.size() > nFeatures) {
            Map<String, Double> featureScores = new HashMap<>();
            
            for (String feature : selectedFeatures) {
                List<String> currentFeatures = new ArrayList<>(selectedFeatures);
                currentFeatures.remove(feature);
                
                double score = evaluateFeatureSet(dataset, currentFeatures, targetFeature);
                featureScores.put(feature, score);
            }
            
            // Remove the feature with the lowest score
            String worstFeature = featureScores.entrySet().stream()
                    .min(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(null);
            
            if (worstFeature != null) {
                selectedFeatures.remove(worstFeature);
            } else {
                break;
            }
        }
        
        return selectedFeatures;
    }
    
    /**
     * L1-based feature selection (Lasso-like)
     */
    public List<String> selectByL1Regularization(Dataset dataset, String targetFeature, double lambda) {
        // Simplified L1 regularization - in practice, this would use actual Lasso regression
        Map<String, Double> featureWeights = new HashMap<>();
        
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double correlation = calculateCorrelation(dataset, feature, targetFeature);
                double weight = Math.max(0, Math.abs(correlation) - lambda);
                featureWeights.put(feature, weight);
            }
        }
        
        return featureWeights.entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
    
    /**
     * Principal Component Analysis (PCA) based selection (simplified)
     */
    public List<String> selectByPCA(Dataset dataset, int nComponents) {
        // Simplified PCA - in practice, this would use actual PCA implementation
        List<String> numericalFeatures = dataset.getNumericalFeatures();
        
        if (numericalFeatures.size() <= nComponents) {
            return numericalFeatures;
        }
        
        // Sort features by variance (simplified approach)
        Map<String, Double> featureVariances = new HashMap<>();
        for (String feature : numericalFeatures) {
            double variance = calculateVariance(dataset, feature);
            featureVariances.put(feature, variance);
        }
        
        return featureVariances.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(nComponents)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
    
    /**
     * Calculates correlation between two features
     */
    private double calculateCorrelation(Dataset dataset, String feature1, String feature2) {
        double mean1 = dataset.calculateMean(feature1);
        double mean2 = dataset.calculateMean(feature2);
        
        double numerator = 0.0;
        double sumSquares1 = 0.0;
        double sumSquares2 = 0.0;
        
        for (DataPoint point : dataset.getDataPoints()) {
            if (point.getFeature(feature1) instanceof Number && point.getFeature(feature2) instanceof Number) {
                double val1 = point.getNumericalFeature(feature1);
                double val2 = point.getNumericalFeature(feature2);
                
                double diff1 = val1 - mean1;
                double diff2 = val2 - mean2;
                
                numerator += diff1 * diff2;
                sumSquares1 += diff1 * diff1;
                sumSquares2 += diff2 * diff2;
            }
        }
        
        if (sumSquares1 == 0 || sumSquares2 == 0) {
            return 0.0;
        }
        
        return numerator / Math.sqrt(sumSquares1 * sumSquares2);
    }
    
    /**
     * Calculates variance of a feature
     */
    private double calculateVariance(Dataset dataset, String featureName) {
        return dataset.calculateVariance(featureName);
    }
    
    /**
     * Calculates mutual information (simplified)
     */
    private double calculateMutualInformation(Dataset dataset, String feature1, String feature2) {
        // Simplified mutual information calculation
        double correlation = Math.abs(calculateCorrelation(dataset, feature1, feature2));
        return -0.5 * Math.log(1 - correlation * correlation);
    }
    
    /**
     * Evaluates a feature set (simplified)
     */
    private double evaluateFeatureSet(Dataset dataset, List<String> features, String targetFeature) {
        // Simplified evaluation - sum of absolute correlations
        return features.stream()
                .mapToDouble(feature -> Math.abs(calculateCorrelation(dataset, feature, targetFeature)))
                .sum();
    }
    
    /**
     * Generates feature selection report
     */
    public void generateFeatureSelectionReport(Dataset dataset, String targetFeature) {
        System.out.println("=== Feature Selection Report ===");
        
        // Correlation-based selection
        List<String> correlationFeatures = selectByCorrelation(dataset, targetFeature, 0.1);
        System.out.printf("Correlation-based features (threshold=0.1): %s%n", correlationFeatures);
        
        // Variance-based selection
        List<String> varianceFeatures = selectByVariance(dataset, 0.01);
        System.out.printf("Variance-based features (threshold=0.01): %s%n", varianceFeatures);
        
        // Mutual information-based selection
        List<String> miFeatures = selectByMutualInformation(dataset, targetFeature, 5);
        System.out.printf("Mutual information-based features (top 5): %s%n", miFeatures);
        
        // Feature importance scores
        System.out.println("\nFeature Importance Scores:");
        for (String feature : dataset.getFeatureNames()) {
            if (!feature.equals(targetFeature)) {
                double correlation = calculateCorrelation(dataset, feature, targetFeature);
                double variance = calculateVariance(dataset, feature);
                System.out.printf("%s: correlation=%.4f, variance=%.4f%n", feature, correlation, variance);
            }
        }
    }
}

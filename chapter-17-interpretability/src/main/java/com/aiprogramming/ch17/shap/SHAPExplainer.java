package com.aiprogramming.ch17.shap;

import com.aiprogramming.ch17.utils.ModelWrapper;

import java.util.*;

/**
 * SHAP (SHapley Additive exPlanations) implementation
 * 
 * SHAP provides both local and global feature importance by calculating
 * Shapley values, which represent the average marginal contribution of
 * each feature across all possible coalitions.
 */
public class SHAPExplainer {
    
    private final ModelWrapper model;
    private final double[][] backgroundData;
    private final String[] featureNames;
    private final int maxSubsetSize;
    private final Random random;
    
    /**
     * Constructor for SHAP explainer
     * 
     * @param model the model to explain
     * @param backgroundData background dataset for reference
     * @param featureNames names of the features
     */
    public SHAPExplainer(ModelWrapper model, double[][] backgroundData, String[] featureNames) {
        this.model = model;
        this.backgroundData = backgroundData;
        this.featureNames = featureNames;
        this.maxSubsetSize = Math.min(10, featureNames.length); // Limit subset size for performance
        this.random = new Random(42); // Fixed seed for reproducibility
    }
    
    /**
     * Calculate SHAP values for a single instance
     * 
     * @param instance the instance to explain
     * @return array of SHAP values for each feature
     */
    public double[] explain(double[] instance) {
        try {
            double[] shapValues = new double[featureNames.length];
            
            // Calculate SHAP values for each feature
            for (int i = 0; i < featureNames.length; i++) {
                shapValues[i] = calculateShapleyValue(instance, i);
            }
            
            return shapValues;
            
        } catch (Exception e) {
            System.err.println("Error calculating SHAP values: " + e.getMessage());
            return new double[featureNames.length];
        }
    }
    
    /**
     * Calculate Shapley value for a specific feature
     * 
     * @param instance the instance to explain
     * @param featureIndex index of the feature
     * @return Shapley value for the feature
     */
    private double calculateShapleyValue(double[] instance, int featureIndex) {
        double shapleyValue = 0.0;
        int totalCoalitions = 0;
        
        // Calculate expected value (baseline)
        double expectedValue = calculateExpectedValue();
        
        // Iterate over all possible subset sizes
        for (int subsetSize = 0; subsetSize <= maxSubsetSize; subsetSize++) {
            List<int[]> coalitions = generateCoalitions(featureNames.length, subsetSize);
            
            for (int[] coalition : coalitions) {
                // Calculate value with feature
                double valueWithFeature = calculateCoalitionValue(instance, coalition, featureIndex, true);
                
                // Calculate value without feature
                double valueWithoutFeature = calculateCoalitionValue(instance, coalition, featureIndex, false);
                
                // Calculate marginal contribution
                double marginalContribution = valueWithFeature - valueWithoutFeature;
                
                // Weight by coalition size
                double weight = calculateCoalitionWeight(coalition.length, featureNames.length);
                
                shapleyValue += weight * marginalContribution;
                totalCoalitions++;
            }
        }
        
        return totalCoalitions > 0 ? shapleyValue / totalCoalitions : 0.0;
    }
    
    /**
     * Calculate the expected value using background data
     */
    private double calculateExpectedValue() {
        double sum = 0.0;
        int count = 0;
        
        for (double[] sample : backgroundData) {
            sum += model.predict(sample);
            count++;
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    /**
     * Generate coalitions of given size
     */
    private List<int[]> generateCoalitions(int totalFeatures, int subsetSize) {
        List<int[]> coalitions = new ArrayList<>();
        
        if (subsetSize == 0) {
            coalitions.add(new int[0]);
            return coalitions;
        }
        
        if (subsetSize > totalFeatures) {
            return coalitions;
        }
        
        // Simple implementation for generating combinations
        // For demonstration purposes, we'll generate a subset of all possible combinations
        if (subsetSize == 1) {
            for (int i = 0; i < totalFeatures; i++) {
                coalitions.add(new int[]{i});
            }
        } else if (subsetSize == 2) {
            for (int i = 0; i < totalFeatures - 1; i++) {
                for (int j = i + 1; j < totalFeatures; j++) {
                    coalitions.add(new int[]{i, j});
                }
            }
        } else {
            // For larger subsets, generate random combinations
            Random rand = new Random(42);
            int numCombinations = Math.min(100, totalFeatures * (totalFeatures - 1) / 2);
            
            for (int k = 0; k < numCombinations; k++) {
                Set<Integer> coalition = new HashSet<>();
                while (coalition.size() < subsetSize) {
                    coalition.add(rand.nextInt(totalFeatures));
                }
                
                int[] coalitionArray = coalition.stream().mapToInt(Integer::intValue).toArray();
                Arrays.sort(coalitionArray);
                coalitions.add(coalitionArray);
            }
        }
        
        return coalitions;
    }
    
    /**
     * Calculate the value of a coalition
     */
    private double calculateCoalitionValue(double[] instance, int[] coalition, int targetFeature, boolean includeTarget) {
        // Create a perturbed instance based on the coalition
        double[] perturbedInstance = createPerturbedInstance(instance, coalition, targetFeature, includeTarget);
        
        // Get prediction for perturbed instance
        return model.predict(perturbedInstance);
    }
    
    /**
     * Create a perturbed instance based on coalition membership
     */
    private double[] createPerturbedInstance(double[] instance, int[] coalition, int targetFeature, boolean includeTarget) {
        double[] perturbed = new double[instance.length];
        
        // Copy original instance
        System.arraycopy(instance, 0, perturbed, 0, instance.length);
        
        // Replace features not in coalition with background values
        Set<Integer> coalitionSet = new HashSet<>();
        for (int feature : coalition) {
            coalitionSet.add(feature);
        }
        
        // Add target feature if requested
        if (includeTarget) {
            coalitionSet.add(targetFeature);
        }
        
        // Replace non-coalition features with background values
        for (int i = 0; i < instance.length; i++) {
            if (!coalitionSet.contains(i)) {
                // Use average of background data for this feature
                perturbed[i] = getBackgroundAverage(i);
            }
        }
        
        return perturbed;
    }
    
    /**
     * Get average value of a feature from background data
     */
    private double getBackgroundAverage(int featureIndex) {
        double sum = 0.0;
        int count = 0;
        
        for (double[] sample : backgroundData) {
            if (featureIndex < sample.length) {
                sum += sample[featureIndex];
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    /**
     * Calculate weight for a coalition based on its size
     */
    private double calculateCoalitionWeight(int coalitionSize, int totalFeatures) {
        if (totalFeatures == 0) {
            return 0.0;
        }
        
        // Weight based on coalition size (smaller coalitions get higher weight)
        return 1.0 / (coalitionSize + 1);
    }
    
    /**
     * Calculate global feature importance using SHAP values
     * 
     * @return array of global importance scores
     */
    public double[] calculateGlobalImportance() {
        double[] globalImportance = new double[featureNames.length];
        
        // Sample instances from background data for efficiency
        int numSamples = Math.min(50, backgroundData.length);
        List<Integer> sampleIndices = new ArrayList<>();
        
        for (int i = 0; i < backgroundData.length; i++) {
            sampleIndices.add(i);
        }
        Collections.shuffle(sampleIndices, random);
        
        // Calculate SHAP values for sample instances
        for (int i = 0; i < numSamples; i++) {
            int index = sampleIndices.get(i);
            double[] instance = backgroundData[index];
            double[] shapValues = explain(instance);
            
            // Accumulate absolute SHAP values
            for (int j = 0; j < featureNames.length; j++) {
                globalImportance[j] += Math.abs(shapValues[j]);
            }
        }
        
        // Normalize by number of samples
        for (int j = 0; j < featureNames.length; j++) {
            globalImportance[j] /= numSamples;
        }
        
        return globalImportance;
    }
    
    /**
     * Calculate SHAP values for multiple instances
     * 
     * @param instances array of instances to explain
     * @return matrix of SHAP values (rows = instances, columns = features)
     */
    public double[][] explainMultiple(double[][] instances) {
        double[][] allShapValues = new double[instances.length][featureNames.length];
        
        for (int i = 0; i < instances.length; i++) {
            allShapValues[i] = explain(instances[i]);
        }
        
        return allShapValues;
    }
    
    /**
     * Get feature names sorted by importance
     * 
     * @param importanceScores importance scores for features
     * @return sorted list of feature names
     */
    public List<String> getFeaturesByImportance(double[] importanceScores) {
        List<FeatureImportance> featureImportances = new ArrayList<>();
        
        for (int i = 0; i < featureNames.length; i++) {
            featureImportances.add(new FeatureImportance(featureNames[i], importanceScores[i]));
        }
        
        // Sort by importance (descending)
        featureImportances.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        List<String> sortedFeatures = new ArrayList<>();
        for (FeatureImportance fi : featureImportances) {
            sortedFeatures.add(fi.name);
        }
        
        return sortedFeatures;
    }
    
    /**
     * Calculate interaction effects between features
     * 
     * @param instance the instance to analyze
     * @return matrix of interaction effects
     */
    public double[][] calculateInteractionEffects(double[] instance) {
        double[][] interactions = new double[featureNames.length][featureNames.length];
        
        for (int i = 0; i < featureNames.length; i++) {
            for (int j = i + 1; j < featureNames.length; j++) {
                double interaction = calculatePairwiseInteraction(instance, i, j);
                interactions[i][j] = interaction;
                interactions[j][i] = interaction;
            }
        }
        
        return interactions;
    }
    
    /**
     * Calculate pairwise interaction between two features
     */
    private double calculatePairwiseInteraction(double[] instance, int feature1, int feature2) {
        // This is a simplified interaction calculation
        // In practice, you would need more sophisticated methods
        
        double[] shapValues = explain(instance);
        double individual1 = shapValues[feature1];
        double individual2 = shapValues[feature2];
        
        // Simplified interaction: product of individual effects
        return individual1 * individual2;
    }
    
    /**
     * Inner class for feature importance ranking
     */
    private static class FeatureImportance {
        final String name;
        final double importance;
        
        FeatureImportance(String name, double importance) {
            this.name = name;
            this.importance = importance;
        }
    }
}

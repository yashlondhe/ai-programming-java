package com.aiprogramming.ch17.fairness;

import com.aiprogramming.ch17.utils.ModelWrapper;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;

/**
 * Fairness metrics implementation for assessing model fairness
 * 
 * This class implements various fairness metrics including demographic parity,
 * equalized odds, equal opportunity, and individual fairness measures.
 */
public class FairnessMetrics {
    
    private final ModelWrapper model;
    private final double[][] data;
    private final Map<String, int[]> sensitiveAttributes; // attribute -> group indices
    private final double[] predictions;
    private final double[] actualLabels;
    
    /**
     * Constructor for fairness metrics
     * 
     * @param model the model to assess
     * @param data the dataset for assessment
     */
    public FairnessMetrics(ModelWrapper model, double[][] data) {
        this.model = model;
        this.data = data;
        this.sensitiveAttributes = new HashMap<>();
        this.predictions = new double[data.length];
        this.actualLabels = new double[data.length];
        
        // Generate predictions and simulate sensitive attributes
        generatePredictionsAndLabels();
        simulateSensitiveAttributes();
    }
    
    /**
     * Generate predictions and simulate actual labels
     */
    private void generatePredictionsAndLabels() {
        for (int i = 0; i < data.length; i++) {
            predictions[i] = model.predict(data[i]);
            
            // Simulate actual labels (in real scenario, these would come from data)
            // For demonstration, we'll use a simple rule based on the first feature
            actualLabels[i] = data[i][0] > 0.5 ? 1.0 : 0.0;
        }
    }
    
    /**
     * Simulate sensitive attributes for demonstration
     */
    private void simulateSensitiveAttributes() {
        Random random = new Random(42);
        
        // Simulate gender (0 = female, 1 = male)
        int[] gender = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            gender[i] = random.nextDouble() < 0.5 ? 0 : 1;
        }
        sensitiveAttributes.put("gender", gender);
        
        // Simulate age groups (0 = young, 1 = middle, 2 = old)
        int[] ageGroup = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            double rand = random.nextDouble();
            if (rand < 0.33) {
                ageGroup[i] = 0;
            } else if (rand < 0.66) {
                ageGroup[i] = 1;
            } else {
                ageGroup[i] = 2;
            }
        }
        sensitiveAttributes.put("age_group", ageGroup);
        
        // Simulate income level (0 = low, 1 = medium, 2 = high)
        int[] incomeLevel = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            double rand = random.nextDouble();
            if (rand < 0.4) {
                incomeLevel[i] = 0;
            } else if (rand < 0.8) {
                incomeLevel[i] = 1;
            } else {
                incomeLevel[i] = 2;
            }
        }
        sensitiveAttributes.put("income_level", incomeLevel);
    }
    
    /**
     * Calculate demographic parity
     * 
     * Demographic parity measures whether the positive prediction rate
     * is similar across different groups.
     * 
     * @param attributeName name of the sensitive attribute
     * @return demographic parity score (closer to 1.0 is more fair)
     */
    public double calculateDemographicParity(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 1.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        Map<Integer, Double> positiveRates = new HashMap<>();
        
        // Calculate positive prediction rate for each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            int group = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            int positiveCount = 0;
            for (int index : indices) {
                if (predictions[index] > 0.5) { // Threshold for positive prediction
                    positiveCount++;
                }
            }
            
            double positiveRate = indices.isEmpty() ? 0.0 : (double) positiveCount / indices.size();
            positiveRates.put(group, positiveRate);
        }
        
        // Calculate fairness score (ratio of minimum to maximum positive rate)
        double minRate = positiveRates.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxRate = positiveRates.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        return maxRate > 0 ? minRate / maxRate : 1.0;
    }
    
    /**
     * Calculate equalized odds
     * 
     * Equalized odds measures whether the true positive rate and false positive rate
     * are similar across different groups.
     * 
     * @param attributeName name of the sensitive attribute
     * @return equalized odds score (closer to 1.0 is more fair)
     */
    public double calculateEqualizedOdds(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 1.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        Map<Integer, Double> tprScores = new HashMap<>();
        Map<Integer, Double> fprScores = new HashMap<>();
        
        // Calculate TPR and FPR for each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            int group = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            int tp = 0, fp = 0, tn = 0, fn = 0;
            
            for (int index : indices) {
                boolean predicted = predictions[index] > 0.5;
                boolean actual = actualLabels[index] > 0.5;
                
                if (predicted && actual) tp++;
                else if (predicted && !actual) fp++;
                else if (!predicted && actual) fn++;
                else tn++;
            }
            
            double tpr = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
            double fpr = (fp + tn) > 0 ? (double) fp / (fp + tn) : 0.0;
            
            tprScores.put(group, tpr);
            fprScores.put(group, fpr);
        }
        
        // Calculate fairness scores
        double tprMin = tprScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double tprMax = tprScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        double fprMin = fprScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double fprMax = fprScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        double tprFairness = tprMax > 0 ? tprMin / tprMax : 1.0;
        double fprFairness = fprMax > 0 ? fprMin / fprMax : 1.0;
        
        return (tprFairness + fprFairness) / 2.0;
    }
    
    /**
     * Calculate equal opportunity
     * 
     * Equal opportunity measures whether the true positive rate
     * is similar across different groups.
     * 
     * @param attributeName name of the sensitive attribute
     * @return equal opportunity score (closer to 1.0 is more fair)
     */
    public double calculateEqualOpportunity(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 1.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        Map<Integer, Double> tprScores = new HashMap<>();
        
        // Calculate TPR for each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            int group = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            int tp = 0, fn = 0;
            
            for (int index : indices) {
                boolean predicted = predictions[index] > 0.5;
                boolean actual = actualLabels[index] > 0.5;
                
                if (predicted && actual) tp++;
                else if (!predicted && actual) fn++;
            }
            
            double tpr = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
            tprScores.put(group, tpr);
        }
        
        // Calculate fairness score
        double minTpr = tprScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxTpr = tprScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        return maxTpr > 0 ? minTpr / maxTpr : 1.0;
    }
    
    /**
     * Calculate individual fairness
     * 
     * Individual fairness measures whether similar individuals receive similar predictions.
     * 
     * @param attributeName name of the sensitive attribute
     * @return individual fairness score (closer to 1.0 is more fair)
     */
    public double calculateIndividualFairness(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 1.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        List<Double> fairnessScores = new ArrayList<>();
        
        // Calculate fairness within each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            List<Integer> indices = entry.getValue();
            
            if (indices.size() < 2) {
                continue;
            }
            
            // Calculate average prediction similarity within group
            double totalSimilarity = 0.0;
            int comparisons = 0;
            
            for (int i = 0; i < indices.size(); i++) {
                for (int j = i + 1; j < indices.size(); j++) {
                    int idx1 = indices.get(i);
                    int idx2 = indices.get(j);
                    
                    // Calculate similarity based on feature values
                    double featureSimilarity = calculateFeatureSimilarity(data[idx1], data[idx2]);
                    
                    // Calculate prediction similarity
                    double predictionSimilarity = 1.0 - Math.abs(predictions[idx1] - predictions[idx2]);
                    
                    // Individual fairness: similar features should have similar predictions
                    double fairness = Math.abs(featureSimilarity - predictionSimilarity);
                    totalSimilarity += fairness;
                    comparisons++;
                }
            }
            
            if (comparisons > 0) {
                fairnessScores.add(1.0 - (totalSimilarity / comparisons));
            }
        }
        
        return fairnessScores.isEmpty() ? 1.0 : 
               fairnessScores.stream().mapToDouble(Double::doubleValue).average().orElse(1.0);
    }
    
    /**
     * Calculate feature similarity between two instances
     */
    private double calculateFeatureSimilarity(double[] instance1, double[] instance2) {
        double distance = 0.0;
        for (int i = 0; i < Math.min(instance1.length, instance2.length); i++) {
            double diff = instance1[i] - instance2[i];
            distance += diff * diff;
        }
        return Math.exp(-Math.sqrt(distance));
    }
    
    /**
     * Get group indices for a sensitive attribute
     */
    private Map<Integer, List<Integer>> getGroupIndices(int[] groups) {
        Map<Integer, List<Integer>> groupIndices = new HashMap<>();
        
        for (int i = 0; i < groups.length; i++) {
            int group = groups[i];
            groupIndices.computeIfAbsent(group, k -> new ArrayList<>()).add(i);
        }
        
        return groupIndices;
    }
    
    /**
     * Generate comprehensive fairness report
     * 
     * @return map of attribute names to their fairness metrics
     */
    public Map<String, Map<String, Double>> generateFairnessReport() {
        Map<String, Map<String, Double>> report = new HashMap<>();
        
        for (String attribute : sensitiveAttributes.keySet()) {
            Map<String, Double> metrics = new HashMap<>();
            
            metrics.put("demographic_parity", calculateDemographicParity(attribute));
            metrics.put("equalized_odds", calculateEqualizedOdds(attribute));
            metrics.put("equal_opportunity", calculateEqualOpportunity(attribute));
            metrics.put("individual_fairness", calculateIndividualFairness(attribute));
            
            report.put(attribute, metrics);
        }
        
        return report;
    }
    
    /**
     * Calculate overall fairness score
     * 
     * @return overall fairness score across all attributes
     */
    public double calculateOverallFairness() {
        Map<String, Map<String, Double>> report = generateFairnessReport();
        
        List<Double> allScores = new ArrayList<>();
        for (Map<String, Double> metrics : report.values()) {
            allScores.addAll(metrics.values());
        }
        
        return allScores.stream().mapToDouble(Double::doubleValue).average().orElse(1.0);
    }
    
    /**
     * Get fairness recommendations based on metrics
     * 
     * @return list of recommendations for improving fairness
     */
    public List<String> getFairnessRecommendations() {
        List<String> recommendations = new ArrayList<>();
        Map<String, Map<String, Double>> report = generateFairnessReport();
        
        for (Map.Entry<String, Map<String, Double>> entry : report.entrySet()) {
            String attribute = entry.getKey();
            Map<String, Double> metrics = entry.getValue();
            
            for (Map.Entry<String, Double> metric : metrics.entrySet()) {
                if (metric.getValue() < 0.8) {
                    recommendations.add(String.format(
                        "Low %s (%.3f) for %s: Consider bias mitigation techniques",
                        metric.getKey(), metric.getValue(), attribute));
                }
            }
        }
        
        if (recommendations.isEmpty()) {
            recommendations.add("Model appears to be fair across all measured attributes");
        }
        
        return recommendations;
    }
}

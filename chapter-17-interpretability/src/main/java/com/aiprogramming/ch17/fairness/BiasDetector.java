package com.aiprogramming.ch17.fairness;

import com.aiprogramming.ch17.utils.ModelWrapper;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;

/**
 * Bias detection implementation for identifying and quantifying bias in ML models
 * 
 * This class implements various bias detection methods including statistical bias,
 * disparate impact, calibration bias, and intersectional bias detection.
 */
public class BiasDetector {
    
    private final ModelWrapper model;
    private final double[][] data;
    private final Map<String, int[]> sensitiveAttributes;
    private final double[] predictions;
    private final double[] actualLabels;
    
    /**
     * Constructor for bias detector
     * 
     * @param model the model to analyze for bias
     * @param data the dataset for analysis
     */
    public BiasDetector(ModelWrapper model, double[][] data) {
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
     * Detect statistical bias
     * 
     * Statistical bias measures whether the distribution of predictions
     * differs significantly across groups.
     * 
     * @param attributeName name of the sensitive attribute
     * @return statistical bias score (higher values indicate more bias)
     */
    public double detectStatisticalBias(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 0.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        Map<Integer, DescriptiveStatistics> groupStats = new HashMap<>();
        
        // Calculate statistics for each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            int group = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            DescriptiveStatistics stats = new DescriptiveStatistics();
            for (int index : indices) {
                stats.addValue(predictions[index]);
            }
            groupStats.put(group, stats);
        }
        
        // Calculate bias as the maximum difference in means between groups
        double maxBias = 0.0;
        List<Integer> groupKeys = new ArrayList<>(groupStats.keySet());
        
        for (int i = 0; i < groupKeys.size(); i++) {
            for (int j = i + 1; j < groupKeys.size(); j++) {
                int group1 = groupKeys.get(i);
                int group2 = groupKeys.get(j);
                
                double mean1 = groupStats.get(group1).getMean();
                double mean2 = groupStats.get(group2).getMean();
                
                double bias = Math.abs(mean1 - mean2);
                maxBias = Math.max(maxBias, bias);
            }
        }
        
        return maxBias;
    }
    
    /**
     * Detect disparate impact
     * 
     * Disparate impact measures whether the ratio of positive outcomes
     * between groups is significantly different from 1.0.
     * 
     * @param attributeName name of the sensitive attribute
     * @return disparate impact score (closer to 1.0 is less biased)
     */
    public double detectDisparateImpact(String attributeName) {
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
                if (predictions[index] > 0.5) {
                    positiveCount++;
                }
            }
            
            double positiveRate = indices.isEmpty() ? 0.0 : (double) positiveCount / indices.size();
            positiveRates.put(group, positiveRate);
        }
        
        // Calculate disparate impact as ratio of minimum to maximum positive rate
        double minRate = positiveRates.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxRate = positiveRates.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        return maxRate > 0 ? minRate / maxRate : 1.0;
    }
    
    /**
     * Detect calibration bias
     * 
     * Calibration bias measures whether the model's predicted probabilities
     * are well-calibrated across different groups.
     * 
     * @param attributeName name of the sensitive attribute
     * @return calibration bias score (lower values indicate better calibration)
     */
    public double detectCalibrationBias(String attributeName) {
        int[] groups = sensitiveAttributes.get(attributeName);
        if (groups == null) {
            return 0.0;
        }
        
        Map<Integer, List<Integer>> groupIndices = getGroupIndices(groups);
        List<Double> calibrationErrors = new ArrayList<>();
        
        // Calculate calibration error for each group
        for (Map.Entry<Integer, List<Integer>> entry : groupIndices.entrySet()) {
            int group = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            if (indices.size() < 10) { // Need sufficient samples
                continue;
            }
            
            // Group predictions into bins
            Map<Integer, List<Integer>> bins = new HashMap<>();
            for (int index : indices) {
                int bin = (int) (predictions[index] * 10); // 10 bins
                bins.computeIfAbsent(bin, k -> new ArrayList<>()).add(index);
            }
            
            // Calculate calibration error for each bin
            for (List<Integer> binIndices : bins.values()) {
                if (binIndices.size() < 5) { // Need sufficient samples per bin
                    continue;
                }
                
                double avgPrediction = binIndices.stream()
                    .mapToDouble(i -> predictions[i])
                    .average()
                    .orElse(0.0);
                
                double actualRate = binIndices.stream()
                    .mapToDouble(i -> actualLabels[i])
                    .average()
                    .orElse(0.0);
                
                double calibrationError = Math.abs(avgPrediction - actualRate);
                calibrationErrors.add(calibrationError);
            }
        }
        
        return calibrationErrors.isEmpty() ? 0.0 : 
               calibrationErrors.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Detect intersectional bias
     * 
     * Intersectional bias considers the combined effect of multiple
     * sensitive attributes.
     * 
     * @param attributeNames array of sensitive attribute names
     * @return intersectional bias score
     */
    public double detectIntersectionalBias(String[] attributeNames) {
        if (attributeNames.length < 2) {
            return 0.0;
        }
        
        // Create intersectional groups
        Map<String, List<Integer>> intersectionalGroups = new HashMap<>();
        
        for (int i = 0; i < data.length; i++) {
            StringBuilder groupKey = new StringBuilder();
            for (String attrName : attributeNames) {
                int[] groups = sensitiveAttributes.get(attrName);
                if (groups != null && i < groups.length) {
                    groupKey.append(attrName).append("_").append(groups[i]).append("_");
                }
            }
            
            String key = groupKey.toString();
            intersectionalGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        
        // Calculate bias across intersectional groups
        List<Double> groupMeans = new ArrayList<>();
        for (List<Integer> indices : intersectionalGroups.values()) {
            if (indices.size() >= 5) { // Need sufficient samples
                double mean = indices.stream()
                    .mapToDouble(j -> predictions[j])
                    .average()
                    .orElse(0.0);
                groupMeans.add(mean);
            }
        }
        
        if (groupMeans.size() < 2) {
            return 0.0;
        }
        
        // Calculate maximum difference between group means
        double maxDiff = 0.0;
        for (int i = 0; i < groupMeans.size(); i++) {
            for (int j = i + 1; j < groupMeans.size(); j++) {
                double diff = Math.abs(groupMeans.get(i) - groupMeans.get(j));
                maxDiff = Math.max(maxDiff, diff);
            }
        }
        
        return maxDiff;
    }
    
    /**
     * Generate comprehensive bias report
     * 
     * @param attributeName name of the sensitive attribute
     * @return map containing various bias metrics
     */
    public Map<String, Object> generateBiasReport(String attributeName) {
        Map<String, Object> report = new HashMap<>();
        
        report.put("statistical_bias", detectStatisticalBias(attributeName));
        report.put("disparate_impact", detectDisparateImpact(attributeName));
        report.put("calibration_bias", detectCalibrationBias(attributeName));
        
        // Determine overall bias level
        double statisticalBias = (Double) report.get("statistical_bias");
        double disparateImpact = (Double) report.get("disparate_impact");
        double calibrationBias = (Double) report.get("calibration_bias");
        
        String biasLevel;
        if (statisticalBias > 0.2 || disparateImpact < 0.8 || calibrationBias > 0.1) {
            biasLevel = "High";
        } else if (statisticalBias > 0.1 || disparateImpact < 0.9 || calibrationBias > 0.05) {
            biasLevel = "Medium";
        } else {
            biasLevel = "Low";
        }
        
        report.put("overall_bias_level", biasLevel);
        
        // Generate recommendations
        List<String> recommendations = new ArrayList<>();
        if (statisticalBias > 0.2) {
            recommendations.add("High statistical bias detected. Consider rebalancing training data.");
        }
        if (disparateImpact < 0.8) {
            recommendations.add("Disparate impact detected. Review feature selection and model training.");
        }
        if (calibrationBias > 0.1) {
            recommendations.add("Calibration bias detected. Consider post-processing calibration.");
        }
        
        if (recommendations.isEmpty()) {
            recommendations.add("No significant bias detected for this attribute.");
        }
        
        report.put("recommendations", recommendations);
        
        return report;
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
     * Calculate overall bias score across all attributes
     * 
     * @return overall bias score
     */
    public double calculateOverallBias() {
        List<Double> biasScores = new ArrayList<>();
        
        for (String attribute : sensitiveAttributes.keySet()) {
            biasScores.add(detectStatisticalBias(attribute));
            biasScores.add(1.0 - detectDisparateImpact(attribute)); // Convert to bias score
            biasScores.add(detectCalibrationBias(attribute));
        }
        
        return biasScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Get bias severity classification
     * 
     * @param biasScore the bias score to classify
     * @return severity classification
     */
    public String classifyBiasSeverity(double biasScore) {
        if (biasScore > 0.3) {
            return "Critical";
        } else if (biasScore > 0.2) {
            return "High";
        } else if (biasScore > 0.1) {
            return "Medium";
        } else if (biasScore > 0.05) {
            return "Low";
        } else {
            return "Minimal";
        }
    }
    
    /**
     * Generate bias mitigation suggestions
     * 
     * @return list of bias mitigation strategies
     */
    public List<String> getBiasMitigationSuggestions() {
        List<String> suggestions = new ArrayList<>();
        
        double overallBias = calculateOverallBias();
        String severity = classifyBiasSeverity(overallBias);
        
        suggestions.add("Overall bias severity: " + severity + " (score: " + String.format("%.3f", overallBias) + ")");
        
        if (overallBias > 0.2) {
            suggestions.add("Consider data preprocessing techniques to reduce bias");
            suggestions.add("Implement fairness-aware training algorithms");
            suggestions.add("Use post-processing bias correction methods");
            suggestions.add("Review feature engineering for potential bias sources");
        } else if (overallBias > 0.1) {
            suggestions.add("Monitor model performance across different groups");
            suggestions.add("Consider rebalancing training data");
            suggestions.add("Implement regular bias audits");
        } else {
            suggestions.add("Continue monitoring for bias in production");
            suggestions.add("Maintain current bias mitigation practices");
        }
        
        return suggestions;
    }
}

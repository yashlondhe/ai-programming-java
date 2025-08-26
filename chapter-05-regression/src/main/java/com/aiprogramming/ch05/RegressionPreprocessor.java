package com.aiprogramming.ch05;

import java.util.*;

/**
 * Data preprocessing utilities for regression
 */
public class RegressionPreprocessor {
    
    /**
     * Normalize features to [0, 1] range
     */
    public List<RegressionDataPoint> normalizeFeatures(List<RegressionDataPoint> data) {
        if (data.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Find min and max for each feature
        Map<String, Double> minValues = new HashMap<>();
        Map<String, Double> maxValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                minValues.put(feature, Math.min(minValues.getOrDefault(feature, Double.MAX_VALUE), value));
                maxValues.put(feature, Math.max(maxValues.getOrDefault(feature, Double.MIN_VALUE), value));
            }
        }
        
        // Normalize each data point
        List<RegressionDataPoint> normalizedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> normalizedFeatures = new HashMap<>();
            
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                double min = minValues.get(feature);
                double max = maxValues.get(feature);
                
                double normalized = (max - min == 0) ? 0.0 : (value - min) / (max - min);
                normalizedFeatures.put(feature, normalized);
            }
            
            normalizedData.add(new RegressionDataPoint(normalizedFeatures, point.getTarget()));
        }
        
        return normalizedData;
    }
    
    /**
     * Standardize features to have mean 0 and standard deviation 1
     */
    public List<RegressionDataPoint> standardizeFeatures(List<RegressionDataPoint> data) {
        if (data.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Calculate means and standard deviations
        Map<String, Double> means = calculateFeatureMeans(data);
        Map<String, Double> stdDevs = calculateFeatureStdDevs(data, means);
        
        // Standardize each data point
        List<RegressionDataPoint> standardizedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> standardizedFeatures = new HashMap<>();
            
            for (String feature : point.getFeatures().keySet()) {
                double value = point.getFeatures().get(feature);
                double mean = means.get(feature);
                double stdDev = stdDevs.get(feature);
                
                double standardized = (stdDev == 0) ? 0.0 : (value - mean) / stdDev;
                standardizedFeatures.put(feature, standardized);
            }
            
            standardizedData.add(new RegressionDataPoint(standardizedFeatures, point.getTarget()));
        }
        
        return standardizedData;
    }
    
    /**
     * Handle missing values using mean imputation
     */
    public List<RegressionDataPoint> handleMissingValues(List<RegressionDataPoint> data) {
        Map<String, Double> featureMeans = calculateFeatureMeans(data);
        
        List<RegressionDataPoint> imputedData = new ArrayList<>();
        for (RegressionDataPoint point : data) {
            Map<String, Double> imputedFeatures = new HashMap<>();
            
            for (String feature : featureMeans.keySet()) {
                double value = point.getFeatures().getOrDefault(feature, featureMeans.get(feature));
                imputedFeatures.put(feature, value);
            }
            
            imputedData.add(new RegressionDataPoint(imputedFeatures, point.getTarget()));
        }
        
        return imputedData;
    }
    
    private Map<String, Double> calculateFeatureMeans(List<RegressionDataPoint> data) {
        Map<String, List<Double>> featureValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                featureValues.computeIfAbsent(feature, k -> new ArrayList<>())
                           .add(point.getFeatures().get(feature));
            }
        }
        
        Map<String, Double> means = new HashMap<>();
        for (String feature : featureValues.keySet()) {
            double mean = featureValues.get(feature).stream()
                                    .mapToDouble(Double::doubleValue)
                                    .average()
                                    .orElse(0.0);
            means.put(feature, mean);
        }
        
        return means;
    }
    
    private Map<String, Double> calculateFeatureStdDevs(List<RegressionDataPoint> data, 
                                                       Map<String, Double> means) {
        Map<String, List<Double>> featureValues = new HashMap<>();
        
        for (RegressionDataPoint point : data) {
            for (String feature : point.getFeatures().keySet()) {
                featureValues.computeIfAbsent(feature, k -> new ArrayList<>())
                           .add(point.getFeatures().get(feature));
            }
        }
        
        Map<String, Double> stdDevs = new HashMap<>();
        for (String feature : featureValues.keySet()) {
            double mean = means.get(feature);
            double variance = featureValues.get(feature).stream()
                                    .mapToDouble(value -> Math.pow(value - mean, 2))
                                    .average()
                                    .orElse(0.0);
            stdDevs.put(feature, Math.sqrt(variance));
        }
        
        return stdDevs;
    }
}
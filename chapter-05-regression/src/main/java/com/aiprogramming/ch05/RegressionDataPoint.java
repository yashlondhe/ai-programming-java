package com.aiprogramming.ch05;

import java.util.Map;
import java.util.HashMap;
import java.util.Set;

/**
 * Represents a single data point for regression with features and target value
 */
public class RegressionDataPoint {
    private final Map<String, Double> features;
    private final double target;
    
    public RegressionDataPoint(Map<String, Double> features, double target) {
        this.features = new HashMap<>(features);
        this.target = target;
    }
    
    public Map<String, Double> getFeatures() {
        return new HashMap<>(features);
    }
    
    public double getTarget() {
        return target;
    }
    
    public Double getFeature(String featureName) {
        return features.get(featureName);
    }
    
    public Set<String> getFeatureNames() {
        return features.keySet();
    }
    
    @Override
    public String toString() {
        return String.format("RegressionDataPoint{features=%s, target=%.3f}", features, target);
    }
}
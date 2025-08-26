package com.aiprogramming.ch04;

import java.util.Map;

/**
 * Represents a data point for classification algorithms.
 * Contains features and a class label.
 */
public class ClassificationDataPoint {
    private final Map<String, Double> features;
    private final String label;
    
    public ClassificationDataPoint(Map<String, Double> features, String label) {
        this.features = features;
        this.label = label;
    }
    
    public Map<String, Double> getFeatures() {
        return features;
    }
    
    public String getLabel() {
        return label;
    }
    
    public double getFeature(String featureName) {
        return features.getOrDefault(featureName, 0.0);
    }
    
    @Override
    public String toString() {
        return "ClassificationDataPoint{" +
                "features=" + features +
                ", label='" + label + '\'' +
                '}';
    }
}

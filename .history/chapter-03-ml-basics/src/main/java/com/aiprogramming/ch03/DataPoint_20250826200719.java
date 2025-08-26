package com.aiprogramming.ch03;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Represents a single data point in a machine learning dataset.
 * Contains features (input variables) and optionally a target value.
 */
public class DataPoint {
    private Map<String, Object> features;
    private Object target;
    
    /**
     * Creates a new data point with features and target
     */
    public DataPoint(Map<String, Object> features, Object target) {
        this.features = new HashMap<>(features);
        this.target = target;
    }
    
    /**
     * Creates a new data point with only features (for unsupervised learning)
     */
    public DataPoint(Map<String, Object> features) {
        this.features = new HashMap<>(features);
        this.target = null;
    }
    
    /**
     * Gets a feature value by name
     */
    public Object getFeature(String featureName) {
        return features.get(featureName);
    }
    
    /**
     * Sets a feature value
     */
    public void setFeature(String featureName, Object value) {
        features.put(featureName, value);
    }
    
    /**
     * Gets all features as a map
     */
    public Map<String, Object> getFeatures() {
        return new HashMap<>(features);
    }
    
    /**
     * Gets the target value
     */
    public Object getTarget() {
        return target;
    }
    
    /**
     * Sets the target value
     */
    public void setTarget(Object target) {
        this.target = target;
    }
    
    /**
     * Checks if the data point has a target value
     */
    public boolean hasTarget() {
        return target != null;
    }
    
    /**
     * Gets a numerical feature value
     */
    public double getNumericalFeature(String featureName) {
        Object value = features.get(featureName);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        throw new IllegalArgumentException("Feature " + featureName + " is not numerical");
    }
    
    /**
     * Gets a categorical feature value
     */
    public String getCategoricalFeature(String featureName) {
        Object value = features.get(featureName);
        if (value instanceof String) {
            return (String) value;
        }
        throw new IllegalArgumentException("Feature " + featureName + " is not categorical");
    }
    
    /**
     * Checks if a feature exists
     */
    public boolean hasFeature(String featureName) {
        return features.containsKey(featureName);
    }
    
    /**
     * Gets the number of features
     */
    public int getFeatureCount() {
        return features.size();
    }
    
    /**
     * Gets all feature names
     */
    public Set<String> getFeatureNames() {
        return new HashSet<>(features.keySet());
    }
    
    /**
     * Creates a copy of this data point
     */
    public DataPoint copy() {
        return new DataPoint(new HashMap<>(features), target);
    }
    
    @Override
    public String toString() {
        return "DataPoint{" +
                "features=" + features +
                ", target=" + target +
                '}';
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        
        DataPoint dataPoint = (DataPoint) o;
        
        if (!features.equals(dataPoint.features)) return false;
        return target != null ? target.equals(dataPoint.target) : dataPoint.target == null;
    }
    
    @Override
    public int hashCode() {
        int result = features.hashCode();
        result = 31 * result + (target != null ? target.hashCode() : 0);
        return result;
    }
}

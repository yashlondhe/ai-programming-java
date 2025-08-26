package com.aiprogramming.ch06;

import java.util.*;

/**
 * Represents a single data point for unsupervised learning algorithms
 */
public class DataPoint {
    private final List<Double> features;
    private final String id;
    private int clusterId;
    
    public DataPoint(List<Double> features) {
        this(features, UUID.randomUUID().toString());
    }
    
    public DataPoint(List<Double> features, String id) {
        this.features = new ArrayList<>(features);
        this.id = id;
        this.clusterId = -1; // Unassigned
    }
    
    public List<Double> getFeatures() {
        return new ArrayList<>(features);
    }
    
    public String getId() {
        return id;
    }
    
    public int getClusterId() {
        return clusterId;
    }
    
    public void setClusterId(int clusterId) {
        this.clusterId = clusterId;
    }
    
    public int getDimension() {
        return features.size();
    }
    
    public double getFeature(int index) {
        if (index < 0 || index >= features.size()) {
            throw new IndexOutOfBoundsException("Feature index out of bounds: " + index);
        }
        return features.get(index);
    }
    
    /**
     * Calculate Euclidean distance to another data point
     */
    public double distanceTo(DataPoint other) {
        if (this.getDimension() != other.getDimension()) {
            throw new IllegalArgumentException("Data points must have the same dimension");
        }
        
        double sum = 0.0;
        for (int i = 0; i < features.size(); i++) {
            double diff = this.features.get(i) - other.features.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Calculate Manhattan distance to another data point
     */
    public double manhattanDistanceTo(DataPoint other) {
        if (this.getDimension() != other.getDimension()) {
            throw new IllegalArgumentException("Data points must have the same dimension");
        }
        
        double sum = 0.0;
        for (int i = 0; i < features.size(); i++) {
            sum += Math.abs(this.features.get(i) - other.features.get(i));
        }
        return sum;
    }
    
    /**
     * Calculate cosine similarity to another data point
     */
    public double cosineSimilarity(DataPoint other) {
        if (this.getDimension() != other.getDimension()) {
            throw new IllegalArgumentException("Data points must have the same dimension");
        }
        
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        
        for (int i = 0; i < features.size(); i++) {
            double a = this.features.get(i);
            double b = other.features.get(i);
            dotProduct += a * b;
            normA += a * a;
            normB += b * b;
        }
        
        if (normA == 0.0 || normB == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    @Override
    public String toString() {
        return String.format("DataPoint{id='%s', features=%s, clusterId=%d}", 
                           id, features, clusterId);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        DataPoint dataPoint = (DataPoint) obj;
        return Objects.equals(id, dataPoint.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}

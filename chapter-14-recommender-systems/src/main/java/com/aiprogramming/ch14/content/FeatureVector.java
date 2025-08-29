package com.aiprogramming.ch14.content;

import java.util.*;

/**
 * Represents a feature vector for content-based filtering.
 * Can handle both categorical and numerical features.
 */
public class FeatureVector {
    private final Map<String, Double> features;
    private final Map<String, String> categoricalFeatures;
    private final String itemId;
    
    /**
     * Constructor for numerical features only.
     * 
     * @param itemId the ID of the item this feature vector represents
     * @param features map of feature names to numerical values
     */
    public FeatureVector(String itemId, Map<String, Double> features) {
        this.itemId = itemId;
        this.features = new HashMap<>(features);
        this.categoricalFeatures = new HashMap<>();
    }
    
    /**
     * Constructor for categorical features only.
     * 
     * @param itemId the ID of the item this feature vector represents
     * @param categoricalFeatures map of feature names to categorical values
     */
    public FeatureVector(String itemId, Map<String, String> categoricalFeatures, boolean isCategorical) {
        this.itemId = itemId;
        this.features = new HashMap<>();
        this.categoricalFeatures = new HashMap<>(categoricalFeatures);
    }
    
    /**
     * Constructor for mixed features.
     * 
     * @param itemId the ID of the item this feature vector represents
     * @param features map of feature names to numerical values
     * @param categoricalFeatures map of feature names to categorical values
     */
    public FeatureVector(String itemId, Map<String, Double> features, Map<String, String> categoricalFeatures) {
        this.itemId = itemId;
        this.features = new HashMap<>(features);
        this.categoricalFeatures = new HashMap<>(categoricalFeatures);
    }
    
    /**
     * Constructor for simple string-based features (e.g., movie genres).
     * 
     * @param itemId the ID of the item
     * @param featureStrings array of feature strings
     */
    public FeatureVector(String itemId, String... featureStrings) {
        this.itemId = itemId;
        this.features = new HashMap<>();
        this.categoricalFeatures = new HashMap<>();
        
        for (String feature : featureStrings) {
            if (feature != null && !feature.trim().isEmpty()) {
                this.categoricalFeatures.put(feature.trim().toLowerCase(), "1");
            }
        }
    }
    
    /**
     * Get the item ID.
     * 
     * @return item ID
     */
    public String getItemId() {
        return itemId;
    }
    
    /**
     * Get a numerical feature value.
     * 
     * @param featureName the name of the feature
     * @return the feature value, or 0.0 if not found
     */
    public double getFeature(String featureName) {
        return features.getOrDefault(featureName, 0.0);
    }
    
    /**
     * Get a categorical feature value.
     * 
     * @param featureName the name of the feature
     * @return the feature value, or null if not found
     */
    public String getCategoricalFeature(String featureName) {
        return categoricalFeatures.get(featureName);
    }
    
    /**
     * Set a numerical feature value.
     * 
     * @param featureName the name of the feature
     * @param value the feature value
     */
    public void setFeature(String featureName, double value) {
        features.put(featureName, value);
    }
    
    /**
     * Set a categorical feature value.
     * 
     * @param featureName the name of the feature
     * @param value the feature value
     */
    public void setCategoricalFeature(String featureName, String value) {
        categoricalFeatures.put(featureName, value);
    }
    
    /**
     * Check if a feature exists.
     * 
     * @param featureName the name of the feature
     * @return true if the feature exists
     */
    public boolean hasFeature(String featureName) {
        return features.containsKey(featureName) || categoricalFeatures.containsKey(featureName);
    }
    
    /**
     * Check if a categorical feature has a specific value.
     * 
     * @param featureName the name of the feature
     * @param value the value to check for
     * @return true if the feature has the specified value
     */
    public boolean hasCategoricalFeature(String featureName, String value) {
        String featureValue = categoricalFeatures.get(featureName);
        return featureValue != null && featureValue.equals(value);
    }
    
    /**
     * Get all numerical feature names.
     * 
     * @return set of numerical feature names
     */
    public Set<String> getNumericalFeatureNames() {
        return new HashSet<>(features.keySet());
    }
    
    /**
     * Get all categorical feature names.
     * 
     * @return set of categorical feature names
     */
    public Set<String> getCategoricalFeatureNames() {
        return new HashSet<>(categoricalFeatures.keySet());
    }
    
    /**
     * Get all feature names (both numerical and categorical).
     * 
     * @return set of all feature names
     */
    public Set<String> getAllFeatureNames() {
        Set<String> allFeatures = new HashSet<>(features.keySet());
        allFeatures.addAll(categoricalFeatures.keySet());
        return allFeatures;
    }
    
    /**
     * Get the magnitude (L2 norm) of the numerical feature vector.
     * 
     * @return magnitude of the feature vector
     */
    public double getMagnitude() {
        return Math.sqrt(features.values().stream()
                .mapToDouble(v -> v * v)
                .sum());
    }
    
    /**
     * Normalize the numerical features to unit length.
     * 
     * @return new normalized feature vector
     */
    public FeatureVector normalize() {
        double magnitude = getMagnitude();
        if (magnitude == 0.0) {
            return this;
        }
        
        Map<String, Double> normalizedFeatures = new HashMap<>();
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            normalizedFeatures.put(entry.getKey(), entry.getValue() / magnitude);
        }
        
        return new FeatureVector(itemId, normalizedFeatures, categoricalFeatures);
    }
    
    /**
     * Calculate cosine similarity with another feature vector.
     * 
     * @param other the other feature vector
     * @return cosine similarity score between 0 and 1
     */
    public double cosineSimilarity(FeatureVector other) {
        if (other == null) return 0.0;
        
        // Calculate dot product of numerical features
        double dotProduct = 0.0;
        Set<String> allFeatures = new HashSet<>(features.keySet());
        allFeatures.addAll(other.features.keySet());
        
        for (String feature : allFeatures) {
            dotProduct += getFeature(feature) * other.getFeature(feature);
        }
        
        // Calculate magnitudes
        double magnitude1 = getMagnitude();
        double magnitude2 = other.getMagnitude();
        
        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (magnitude1 * magnitude2);
    }
    
    /**
     * Calculate Jaccard similarity for categorical features.
     * 
     * @param other the other feature vector
     * @return Jaccard similarity score between 0 and 1
     */
    public double jaccardSimilarity(FeatureVector other) {
        if (other == null) return 0.0;
        
        Set<String> features1 = new HashSet<>(categoricalFeatures.keySet());
        Set<String> features2 = new HashSet<>(other.categoricalFeatures.keySet());
        
        if (features1.isEmpty() && features2.isEmpty()) {
            return 1.0;
        }
        
        Set<String> intersection = new HashSet<>(features1);
        intersection.retainAll(features2);
        
        Set<String> union = new HashSet<>(features1);
        union.addAll(features2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    /**
     * Calculate overall similarity combining numerical and categorical features.
     * 
     * @param other the other feature vector
     * @param numericalWeight weight for numerical similarity (0-1)
     * @return combined similarity score
     */
    public double combinedSimilarity(FeatureVector other, double numericalWeight) {
        double categoricalWeight = 1.0 - numericalWeight;
        
        double numericalSim = cosineSimilarity(other);
        double categoricalSim = jaccardSimilarity(other);
        
        return numericalWeight * numericalSim + categoricalWeight * categoricalSim;
    }
    
    /**
     * Get the number of features.
     * 
     * @return total number of features
     */
    public int getFeatureCount() {
        return features.size() + categoricalFeatures.size();
    }
    
    /**
     * Create a feature vector from a list of genres.
     * 
     * @param itemId the item ID
     * @param genres list of genres
     * @return feature vector with genres as categorical features
     */
    public static FeatureVector fromGenres(String itemId, List<String> genres) {
        Map<String, String> genreFeatures = new HashMap<>();
        for (String genre : genres) {
            if (genre != null && !genre.trim().isEmpty()) {
                genreFeatures.put(genre.trim().toLowerCase(), "1");
            }
        }
        return new FeatureVector(itemId, new HashMap<>(), genreFeatures);
    }
    
    /**
     * Create a feature vector from a map of numerical features.
     * 
     * @param itemId the item ID
     * @param features map of feature names to values
     * @return feature vector with numerical features
     */
    public static FeatureVector fromNumericalFeatures(String itemId, Map<String, Double> features) {
        return new FeatureVector(itemId, features);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        FeatureVector that = (FeatureVector) obj;
        return Objects.equals(itemId, that.itemId) &&
               Objects.equals(features, that.features) &&
               Objects.equals(categoricalFeatures, that.categoricalFeatures);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(itemId, features, categoricalFeatures);
    }
    
    @Override
    public String toString() {
        return String.format("FeatureVector{itemId='%s', numericalFeatures=%s, categoricalFeatures=%s}",
                           itemId, features, categoricalFeatures);
    }
}

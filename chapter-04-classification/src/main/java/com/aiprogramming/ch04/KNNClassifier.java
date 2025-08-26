package com.aiprogramming.ch04;

import java.util.*;
import java.util.stream.Collectors;

/**
 * K-Nearest Neighbors (KNN) Classifier
 * 
 * A simple but effective classification algorithm that classifies
 * new data points based on the majority class of their k nearest neighbors.
 */
public class KNNClassifier implements Classifier {
    
    private final int k;
    private List<ClassificationDataPoint> trainingData;
    private Set<String> featureNames;
    
    public KNNClassifier(int k) {
        this.k = k;
        this.trainingData = new ArrayList<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        this.trainingData = new ArrayList<>(trainingData);
        
        // Extract feature names from training data
        this.featureNames = new HashSet<>();
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (trainingData.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        // Find k nearest neighbors
        List<ClassificationDataPoint> neighbors = findKNearestNeighbors(features);
        
        // Count votes for each class
        Map<String, Long> classVotes = neighbors.stream()
                .collect(Collectors.groupingBy(
                    ClassificationDataPoint::getLabel,
                    Collectors.counting()
                ));
        
        // Return the class with the most votes
        return classVotes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Finds the k nearest neighbors using Euclidean distance
     */
    private List<ClassificationDataPoint> findKNearestNeighbors(Map<String, Double> features) {
        // Calculate distances to all training points
        List<DistancePoint> distances = new ArrayList<>();
        
        for (ClassificationDataPoint trainingPoint : trainingData) {
            double distance = calculateEuclideanDistance(features, trainingPoint.getFeatures());
            distances.add(new DistancePoint(distance, trainingPoint));
        }
        
        // Sort by distance and return k nearest
        return distances.stream()
                .sorted(Comparator.comparingDouble(DistancePoint::getDistance))
                .limit(k)
                .map(DistancePoint::getDataPoint)
                .collect(Collectors.toList());
    }
    
    /**
     * Calculates Euclidean distance between two feature vectors
     */
    private double calculateEuclideanDistance(Map<String, Double> features1, Map<String, Double> features2) {
        double sumSquaredDiff = 0.0;
        
        // Use all feature names from training data
        for (String featureName : featureNames) {
            double val1 = features1.getOrDefault(featureName, 0.0);
            double val2 = features2.getOrDefault(featureName, 0.0);
            double diff = val1 - val2;
            sumSquaredDiff += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiff);
    }
    
    /**
     * Helper class to store distance and data point together
     */
    private static class DistancePoint {
        private final double distance;
        private final ClassificationDataPoint dataPoint;
        
        public DistancePoint(double distance, ClassificationDataPoint dataPoint) {
            this.distance = distance;
            this.dataPoint = dataPoint;
        }
        
        public double getDistance() {
            return distance;
        }
        
        public ClassificationDataPoint getDataPoint() {
            return dataPoint;
        }
    }
    
    public int getK() {
        return k;
    }
    
    public int getTrainingDataSize() {
        return trainingData.size();
    }
}

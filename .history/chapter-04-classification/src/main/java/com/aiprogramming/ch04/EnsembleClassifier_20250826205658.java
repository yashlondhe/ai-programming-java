package com.aiprogramming.ch04;

import java.util.*;

/**
 * Ensemble classifier that combines multiple classifiers using voting.
 * Implements majority voting for classification.
 */
public class EnsembleClassifier implements Classifier {
    
    private final List<Classifier> classifiers;
    private boolean isTrained = false;
    
    public EnsembleClassifier() {
        this.classifiers = new ArrayList<>();
    }
    
    /**
     * Adds a classifier to the ensemble
     */
    public void addClassifier(Classifier classifier) {
        classifiers.add(classifier);
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (classifiers.isEmpty()) {
            throw new IllegalStateException("No classifiers added to ensemble");
        }
        
        // Train each classifier
        for (Classifier classifier : classifiers) {
            classifier.train(trainingData);
        }
        
        isTrained = true;
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (!isTrained) {
            throw new IllegalStateException("Ensemble must be trained before making predictions");
        }
        
        // Collect predictions from all classifiers
        Map<String, Integer> voteCounts = new HashMap<>();
        
        for (Classifier classifier : classifiers) {
            String prediction = classifier.predict(features);
            voteCounts.put(prediction, voteCounts.getOrDefault(prediction, 0) + 1);
        }
        
        // Return the class with the most votes
        return voteCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Gets the number of classifiers in the ensemble
     */
    public int getClassifierCount() {
        return classifiers.size();
    }
    
    /**
     * Gets the list of classifiers
     */
    public List<Classifier> getClassifiers() {
        return new ArrayList<>(classifiers);
    }
}

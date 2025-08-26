package com.aiprogramming.ch04;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Naive Bayes Classifier
 * 
 * A probabilistic classifier based on Bayes' theorem with an assumption
 * of conditional independence between features. Uses Gaussian distribution
 * for continuous features.
 */
public class NaiveBayesClassifier implements Classifier {
    
    private Map<String, ClassProbabilities> classProbabilities;
    private Set<String> featureNames;
    private int totalSamples;
    
    public NaiveBayesClassifier() {
        this.classProbabilities = new HashMap<>();
        this.featureNames = new HashSet<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        this.totalSamples = trainingData.size();
        this.classProbabilities.clear();
        this.featureNames.clear();
        
        // Extract feature names
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
        
        // Group data by class
        Map<String, List<ClassificationDataPoint>> dataByClass = trainingData.stream()
                .collect(Collectors.groupingBy(ClassificationDataPoint::getLabel));
        
        // Calculate probabilities for each class
        for (Map.Entry<String, List<ClassificationDataPoint>> entry : dataByClass.entrySet()) {
            String className = entry.getKey();
            List<ClassificationDataPoint> classData = entry.getValue();
            
            ClassProbabilities classProb = new ClassProbabilities(className, classData.size(), totalSamples);
            
            // Calculate feature probabilities for this class
            for (String featureName : featureNames) {
                List<Double> featureValues = classData.stream()
                        .map(point -> point.getFeature(featureName))
                        .collect(Collectors.toList());
                
                FeatureStatistics stats = calculateFeatureStatistics(featureValues);
                classProb.addFeatureStatistics(featureName, stats);
            }
            
            classProbabilities.put(className, classProb);
        }
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (classProbabilities.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        String bestClass = null;
        double bestProbability = Double.NEGATIVE_INFINITY;
        
        // Calculate probability for each class
        for (ClassProbabilities classProb : classProbabilities.values()) {
            double probability = calculateClassProbability(features, classProb);
            
            if (probability > bestProbability) {
                bestProbability = probability;
                bestClass = classProb.getClassName();
            }
        }
        
        return bestClass != null ? bestClass : "unknown";
    }
    
    /**
     * Calculates the probability of a class given the features
     */
    private double calculateClassProbability(Map<String, Double> features, ClassProbabilities classProb) {
        // Start with prior probability (log to avoid numerical underflow)
        double logProbability = Math.log(classProb.getPriorProbability());
        
        // Add likelihood for each feature
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            FeatureStatistics stats = classProb.getFeatureStatistics(featureName);
            
            if (stats != null) {
                double likelihood = calculateGaussianLikelihood(featureValue, stats);
                logProbability += Math.log(likelihood);
            }
        }
        
        return logProbability;
    }
    
    /**
     * Calculates Gaussian likelihood for a feature value
     */
    private double calculateGaussianLikelihood(double value, FeatureStatistics stats) {
        double mean = stats.getMean();
        double stdDev = stats.getStandardDeviation();
        
        if (stdDev == 0) {
            return value == mean ? 1.0 : 0.0;
        }
        
        double exponent = -0.5 * Math.pow((value - mean) / stdDev, 2);
        return (1.0 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    }
    
    /**
     * Calculates mean and standard deviation for a list of values
     */
    private FeatureStatistics calculateFeatureStatistics(List<Double> values) {
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
        
        double stdDev = Math.sqrt(variance);
        
        return new FeatureStatistics(mean, stdDev);
    }
    
    /**
     * Class to store probabilities for a specific class
     */
    private static class ClassProbabilities {
        private final String className;
        private final double priorProbability;
        private final Map<String, FeatureStatistics> featureStats;
        
        public ClassProbabilities(String className, int classCount, int totalCount) {
            this.className = className;
            this.priorProbability = (double) classCount / totalCount;
            this.featureStats = new HashMap<>();
        }
        
        public void addFeatureStatistics(String featureName, FeatureStatistics stats) {
            featureStats.put(featureName, stats);
        }
        
        public String getClassName() {
            return className;
        }
        
        public double getPriorProbability() {
            return priorProbability;
        }
        
        public FeatureStatistics getFeatureStatistics(String featureName) {
            return featureStats.get(featureName);
        }
    }
    
    /**
     * Class to store mean and standard deviation for a feature
     */
    private static class FeatureStatistics {
        private final double mean;
        private final double standardDeviation;
        
        public FeatureStatistics(double mean, double standardDeviation) {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
        
        public double getMean() {
            return mean;
        }
        
        public double getStandardDeviation() {
            return standardDeviation;
        }
    }
    
    public Map<String, Double> getClassPriorProbabilities() {
        return classProbabilities.entrySet().stream()
                .collect(Collectors.toMap(
                    Map.Entry::getKey,
                    entry -> entry.getValue().getPriorProbability()
                ));
    }
}

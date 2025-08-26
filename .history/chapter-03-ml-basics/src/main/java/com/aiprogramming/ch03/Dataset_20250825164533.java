package com.aiprogramming.ch03;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a collection of data points for machine learning.
 * Provides utility methods for data manipulation and analysis.
 */
public class Dataset {
    private List<DataPoint> dataPoints;
    
    /**
     * Creates a new dataset from a list of data points
     */
    public Dataset(List<DataPoint> dataPoints) {
        this.dataPoints = new ArrayList<>(dataPoints);
    }
    
    /**
     * Creates an empty dataset
     */
    public Dataset() {
        this.dataPoints = new ArrayList<>();
    }
    
    /**
     * Adds a data point to the dataset
     */
    public void addDataPoint(DataPoint dataPoint) {
        dataPoints.add(dataPoint);
    }
    
    /**
     * Gets all data points
     */
    public List<DataPoint> getDataPoints() {
        return new ArrayList<>(dataPoints);
    }
    
    /**
     * Gets the size of the dataset
     */
    public int size() {
        return dataPoints.size();
    }
    
    /**
     * Checks if the dataset is empty
     */
    public boolean isEmpty() {
        return dataPoints.isEmpty();
    }
    
    /**
     * Gets all feature names from the dataset
     */
    public Set<String> getFeatureNames() {
        if (dataPoints.isEmpty()) {
            return new HashSet<>();
        }
        return new HashSet<>(dataPoints.get(0).getFeatures().keySet());
    }
    
    /**
     * Gets numerical feature names
     */
    public List<String> getNumericalFeatures() {
        if (dataPoints.isEmpty()) {
            return new ArrayList<>();
        }
        
        Set<String> numericalFeatures = new HashSet<>();
        DataPoint firstPoint = dataPoints.get(0);
        
        for (String featureName : firstPoint.getFeatureNames()) {
            if (firstPoint.getFeature(featureName) instanceof Number) {
                numericalFeatures.add(featureName);
            }
        }
        
        return new ArrayList<>(numericalFeatures);
    }
    
    /**
     * Gets categorical feature names
     */
    public List<String> getCategoricalFeatures() {
        if (dataPoints.isEmpty()) {
            return new ArrayList<>();
        }
        
        Set<String> categoricalFeatures = new HashSet<>();
        DataPoint firstPoint = dataPoints.get(0);
        
        for (String featureName : firstPoint.getFeatureNames()) {
            if (firstPoint.getFeature(featureName) instanceof String) {
                categoricalFeatures.add(featureName);
            }
        }
        
        return new ArrayList<>(categoricalFeatures);
    }
    
    /**
     * Gets all target values
     */
    public List<Object> getTargetValues() {
        return dataPoints.stream()
                .filter(DataPoint::hasTarget)
                .map(DataPoint::getTarget)
                .collect(Collectors.toList());
    }
    
    /**
     * Gets all labels (for classification)
     */
    public List<Object> getLabels() {
        return getTargetValues();
    }
    
    /**
     * Calculates the mean of a numerical feature
     */
    public double calculateMean(String featureName) {
        return dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> point.getNumericalFeature(featureName))
                .average()
                .orElse(0.0);
    }
    
    /**
     * Calculates the median of a numerical feature
     */
    public double calculateMedian(String featureName) {
        List<Double> values = dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .map(point -> point.getNumericalFeature(featureName))
                .sorted()
                .collect(Collectors.toList());
        
        if (values.isEmpty()) {
            return 0.0;
        }
        
        int size = values.size();
        if (size % 2 == 0) {
            return (values.get(size / 2 - 1) + values.get(size / 2)) / 2.0;
        } else {
            return values.get(size / 2);
        }
    }
    
    /**
     * Calculates the standard deviation of a numerical feature
     */
    public double calculateStandardDeviation(String featureName) {
        double mean = calculateMean(featureName);
        double variance = dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> Math.pow(point.getNumericalFeature(featureName) - mean, 2))
                .average()
                .orElse(0.0);
        
        return Math.sqrt(variance);
    }
    
    /**
     * Calculates the variance of a numerical feature
     */
    public double calculateVariance(String featureName) {
        double mean = calculateMean(featureName);
        return dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> Math.pow(point.getNumericalFeature(featureName) - mean, 2))
                .average()
                .orElse(0.0);
    }
    
    /**
     * Gets the minimum value of a numerical feature
     */
    public double getMin(String featureName) {
        return dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> point.getNumericalFeature(featureName))
                .min()
                .orElse(0.0);
    }
    
    /**
     * Gets the maximum value of a numerical feature
     */
    public double getMax(String featureName) {
        return dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> point.getNumericalFeature(featureName))
                .max()
                .orElse(0.0);
    }
    
    /**
     * Calculates a percentile of a numerical feature
     */
    public double calculatePercentile(String featureName, double percentile) {
        List<Double> values = dataPoints.stream()
                .filter(point -> point.hasFeature(featureName) && point.getFeature(featureName) instanceof Number)
                .map(point -> point.getNumericalFeature(featureName))
                .sorted()
                .collect(Collectors.toList());
        
        if (values.isEmpty()) {
            return 0.0;
        }
        
        int index = (int) Math.ceil((percentile / 100.0) * values.size()) - 1;
        return values.get(Math.max(0, index));
    }
    
    /**
     * Filters the dataset based on a predicate
     */
    public Dataset filter(java.util.function.Predicate<DataPoint> predicate) {
        List<DataPoint> filtered = dataPoints.stream()
                .filter(predicate)
                .collect(Collectors.toList());
        return new Dataset(filtered);
    }
    
    /**
     * Maps each data point using a function
     */
    public Dataset map(java.util.function.Function<DataPoint, DataPoint> mapper) {
        List<DataPoint> mapped = dataPoints.stream()
                .map(mapper)
                .collect(Collectors.toList());
        return new Dataset(mapped);
    }
    
    /**
     * Splits the dataset into training, validation, and test sets
     */
    public DatasetSplit split(double trainingRatio, double validationRatio, double testRatio) {
        if (Math.abs(trainingRatio + validationRatio + testRatio - 1.0) > 1e-6) {
            throw new IllegalArgumentException("Ratios must sum to 1.0");
        }
        
        // Shuffle the data
        List<DataPoint> shuffled = new ArrayList<>(dataPoints);
        Collections.shuffle(shuffled, new Random(42));
        
        int totalSize = shuffled.size();
        int trainingSize = (int) (totalSize * trainingRatio);
        int validationSize = (int) (totalSize * validationRatio);
        
        List<DataPoint> trainingData = shuffled.subList(0, trainingSize);
        List<DataPoint> validationData = shuffled.subList(trainingSize, trainingSize + validationSize);
        List<DataPoint> testData = shuffled.subList(trainingSize + validationSize, totalSize);
        
        return new DatasetSplit(
                new Dataset(trainingData),
                new Dataset(validationData),
                new Dataset(testData)
        );
    }
    
    /**
     * Creates a copy of this dataset
     */
    public Dataset copy() {
        List<DataPoint> copied = dataPoints.stream()
                .map(DataPoint::copy)
                .collect(Collectors.toList());
        return new Dataset(copied);
    }
    
    /**
     * Checks if the dataset has missing values
     */
    public boolean hasMissingValues() {
        return dataPoints.stream()
                .anyMatch(point -> point.getFeatures().values().stream()
                        .anyMatch(Objects::isNull));
    }
    
    /**
     * Gets the number of missing values for a specific feature
     */
    public long getMissingValueCount(String featureName) {
        return dataPoints.stream()
                .filter(point -> point.getFeature(featureName) == null)
                .count();
    }
    
    @Override
    public String toString() {
        return "Dataset{" +
                "size=" + size() +
                ", features=" + getFeatureNames() +
                '}';
    }
}

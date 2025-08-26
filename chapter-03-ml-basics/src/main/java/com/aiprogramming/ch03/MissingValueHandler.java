package com.aiprogramming.ch03;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Handles missing values in datasets using various imputation strategies.
 */
public class MissingValueHandler {
    
    /**
     * Removes data points with any missing values
     */
    public Dataset removeMissingValues(Dataset dataset) {
        return dataset.filter(point -> !hasMissingValues(point));
    }
    
    /**
     * Fills missing values with the mean of the feature
     */
    public Dataset fillWithMean(Dataset dataset, String featureName) {
        double mean = dataset.calculateMean(featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, mean);
            }
            return point;
        });
    }
    
    /**
     * Fills missing values with the median of the feature
     */
    public Dataset fillWithMedian(Dataset dataset, String featureName) {
        double median = dataset.calculateMedian(featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, median);
            }
            return point;
        });
    }
    
    /**
     * Fills missing values with the mode of the feature
     */
    public Dataset fillWithMode(Dataset dataset, String featureName) {
        Object mode = calculateMode(dataset, featureName);
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, mode);
            }
            return point;
        });
    }
    
    /**
     * Fills missing values with a constant value
     */
    public Dataset fillWithConstant(Dataset dataset, String featureName, Object constantValue) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) == null) {
                point.setFeature(featureName, constantValue);
            }
            return point;
        });
    }
    
    /**
     * Fills missing values using forward fill (for time series data)
     */
    public Dataset fillWithForwardFill(Dataset dataset, String featureName) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        Object lastValue = null;
        
        for (DataPoint point : dataPoints) {
            if (point.getFeature(featureName) != null) {
                lastValue = point.getFeature(featureName);
            } else if (lastValue != null) {
                point.setFeature(featureName, lastValue);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Fills missing values using backward fill (for time series data)
     */
    public Dataset fillWithBackwardFill(Dataset dataset, String featureName) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        Object nextValue = null;
        
        // Find the last non-null value
        for (int i = dataPoints.size() - 1; i >= 0; i--) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeature(featureName) != null) {
                nextValue = point.getFeature(featureName);
                break;
            }
        }
        
        // Fill backwards
        for (int i = dataPoints.size() - 1; i >= 0; i--) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeature(featureName) != null) {
                nextValue = point.getFeature(featureName);
            } else if (nextValue != null) {
                point.setFeature(featureName, nextValue);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Fills missing values using interpolation (for numerical features)
     */
    public Dataset fillWithInterpolation(Dataset dataset, String featureName) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        List<Double> values = new ArrayList<>();
        List<Integer> indices = new ArrayList<>();
        
        // Collect non-null values and their indices
        for (int i = 0; i < dataPoints.size(); i++) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeature(featureName) instanceof Number) {
                values.add(point.getNumericalFeature(featureName));
                indices.add(i);
            }
        }
        
        // Interpolate missing values
        for (int i = 0; i < dataPoints.size(); i++) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeature(featureName) == null) {
                double interpolatedValue = interpolateValue(i, indices, values);
                point.setFeature(featureName, interpolatedValue);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Fills missing values using K-Nearest Neighbors
     */
    public Dataset fillWithKNN(Dataset dataset, String featureName, int k) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        for (int i = 0; i < dataPoints.size(); i++) {
            DataPoint point = dataPoints.get(i);
            if (point.getFeature(featureName) == null) {
                double knnValue = findKNNValue(point, dataPoints, featureName, k, i);
                point.setFeature(featureName, knnValue);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Checks if a data point has any missing values
     */
    private boolean hasMissingValues(DataPoint point) {
        return point.getFeatures().values().stream().anyMatch(Objects::isNull);
    }
    
    /**
     * Calculates the mode of a feature
     */
    private Object calculateMode(Dataset dataset, String featureName) {
        Map<Object, Long> frequency = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) != null)
                .collect(Collectors.groupingBy(
                        point -> point.getFeature(featureName),
                        Collectors.counting()
                ));
        
        return frequency.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }
    
    /**
     * Interpolates a value at a given index
     */
    private double interpolateValue(int targetIndex, List<Integer> indices, List<Double> values) {
        if (indices.isEmpty()) {
            return 0.0;
        }
        
        // Find the closest indices
        int leftIndex = -1;
        int rightIndex = -1;
        
        for (int i = 0; i < indices.size(); i++) {
            if (indices.get(i) <= targetIndex) {
                leftIndex = i;
            } else {
                rightIndex = i;
                break;
            }
        }
        
        // Handle edge cases
        if (leftIndex == -1) {
            return values.get(rightIndex);
        }
        if (rightIndex == -1) {
            return values.get(leftIndex);
        }
        
        // Linear interpolation
        int leftPos = indices.get(leftIndex);
        int rightPos = indices.get(rightIndex);
        double leftVal = values.get(leftIndex);
        double rightVal = values.get(rightIndex);
        
        double weight = (double) (targetIndex - leftPos) / (rightPos - leftPos);
        return leftVal + weight * (rightVal - leftVal);
    }
    
    /**
     * Finds the K-Nearest Neighbors value for a data point
     */
    private double findKNNValue(DataPoint target, List<DataPoint> dataPoints, 
                               String featureName, int k, int excludeIndex) {
        List<DistancePoint> distances = new ArrayList<>();
        
        for (int i = 0; i < dataPoints.size(); i++) {
            if (i != excludeIndex && dataPoints.get(i).getFeature(featureName) != null) {
                double distance = calculateDistance(target, dataPoints.get(i));
                distances.add(new DistancePoint(distance, dataPoints.get(i).getNumericalFeature(featureName)));
            }
        }
        
        // Sort by distance and take top k
        distances.sort(Comparator.comparingDouble(DistancePoint::getDistance));
        List<DistancePoint> nearest = distances.subList(0, Math.min(k, distances.size()));
        
        // Calculate weighted average
        double sum = 0.0;
        double totalWeight = 0.0;
        
        for (DistancePoint dp : nearest) {
            double weight = 1.0 / (dp.getDistance() + 1e-6); // Add small epsilon to avoid division by zero
            sum += weight * dp.getValue();
            totalWeight += weight;
        }
        
        return totalWeight > 0 ? sum / totalWeight : 0.0;
    }
    
    /**
     * Calculates Euclidean distance between two data points
     */
    private double calculateDistance(DataPoint p1, DataPoint p2) {
        double sum = 0.0;
        Set<String> features = p1.getFeatureNames();
        
        for (String feature : features) {
            Object val1 = p1.getFeature(feature);
            Object val2 = p2.getFeature(feature);
            
            if (val1 instanceof Number && val2 instanceof Number) {
                double diff = ((Number) val1).doubleValue() - ((Number) val2).doubleValue();
                sum += diff * diff;
            }
        }
        
        return Math.sqrt(sum);
    }
    
    /**
     * Helper class for KNN calculations
     */
    private static class DistancePoint {
        private final double distance;
        private final double value;
        
        public DistancePoint(double distance, double value) {
            this.distance = distance;
            this.value = value;
        }
        
        public double getDistance() {
            return distance;
        }
        
        public double getValue() {
            return value;
        }
    }
}

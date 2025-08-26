package com.aiprogramming.ch03;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Provides methods for detecting outliers in datasets.
 */
public class OutlierDetector {
    
    /**
     * Z-score method for outlier detection
     */
    public List<DataPoint> detectOutliersZScore(Dataset dataset, String featureName, double threshold) {
        double mean = dataset.calculateMean(featureName);
        double std = dataset.calculateStandardDeviation(featureName);
        
        if (std == 0) {
            return new ArrayList<>(); // No outliers if no variance
        }
        
        return dataset.getDataPoints().stream()
                .filter(point -> {
                    if (point.getFeature(featureName) instanceof Number) {
                        double value = point.getNumericalFeature(featureName);
                        double zScore = Math.abs((value - mean) / std);
                        return zScore > threshold;
                    }
                    return false;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * IQR method for outlier detection
     */
    public List<DataPoint> detectOutliersIQR(Dataset dataset, String featureName, double multiplier) {
        double q1 = dataset.calculatePercentile(featureName, 25);
        double q3 = dataset.calculatePercentile(featureName, 75);
        double iqr = q3 - q1;
        
        double lowerBound = q1 - multiplier * iqr;
        double upperBound = q3 + multiplier * iqr;
        
        return dataset.getDataPoints().stream()
                .filter(point -> {
                    if (point.getFeature(featureName) instanceof Number) {
                        double value = point.getNumericalFeature(featureName);
                        return value < lowerBound || value > upperBound;
                    }
                    return false;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * Modified Z-score method (more robust to outliers)
     */
    public List<DataPoint> detectOutliersModifiedZScore(Dataset dataset, String featureName, double threshold) {
        double median = dataset.calculateMedian(featureName);
        double mad = calculateMedianAbsoluteDeviation(dataset, featureName);
        
        if (mad == 0) {
            return new ArrayList<>();
        }
        
        return dataset.getDataPoints().stream()
                .filter(point -> {
                    if (point.getFeature(featureName) instanceof Number) {
                        double value = point.getNumericalFeature(featureName);
                        double modifiedZScore = 0.6745 * Math.abs(value - median) / mad;
                        return modifiedZScore > threshold;
                    }
                    return false;
                })
                .collect(Collectors.toList());
    }
    
    /**
     * Isolation Forest method (simplified implementation)
     */
    public List<DataPoint> detectOutliersIsolationForest(Dataset dataset, String featureName, double contamination) {
        // Simplified implementation - in practice, this would be more complex
        List<DataPoint> dataPoints = dataset.getDataPoints();
        int numOutliers = (int) (dataPoints.size() * contamination);
        
        // Sort by feature value and mark extremes as outliers
        List<DataPoint> sorted = dataPoints.stream()
                .filter(point -> point.getFeature(featureName) instanceof Number)
                .sorted(Comparator.comparingDouble(point -> point.getNumericalFeature(featureName)))
                .collect(Collectors.toList());
        
        List<DataPoint> outliers = new ArrayList<>();
        for (int i = 0; i < numOutliers / 2; i++) {
            outliers.add(sorted.get(i)); // Lower outliers
            outliers.add(sorted.get(sorted.size() - 1 - i)); // Upper outliers
        }
        
        return outliers;
    }
    
    /**
     * Local Outlier Factor (LOF) method (simplified implementation)
     */
    public List<DataPoint> detectOutliersLOF(Dataset dataset, String featureName, int k, double threshold) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        List<DataPoint> outliers = new ArrayList<>();
        
        for (DataPoint point : dataPoints) {
            if (point.getFeature(featureName) instanceof Number) {
                double lof = calculateLOF(point, dataPoints, featureName, k);
                if (lof > threshold) {
                    outliers.add(point);
                }
            }
        }
        
        return outliers;
    }
    
    /**
     * Removes outliers from the dataset
     */
    public Dataset removeOutliers(Dataset dataset, String featureName, String method, double threshold) {
        List<DataPoint> outliers;
        
        switch (method.toLowerCase()) {
            case "zscore":
                outliers = detectOutliersZScore(dataset, featureName, threshold);
                break;
            case "iqr":
                outliers = detectOutliersIQR(dataset, featureName, threshold);
                break;
            case "modified_zscore":
                outliers = detectOutliersModifiedZScore(dataset, featureName, threshold);
                break;
            default:
                throw new IllegalArgumentException("Unknown outlier detection method: " + method);
        }
        
        Set<DataPoint> outlierSet = new HashSet<>(outliers);
        List<DataPoint> filtered = dataset.getDataPoints().stream()
                .filter(point -> !outlierSet.contains(point))
                .collect(Collectors.toList());
        
        return new Dataset(filtered);
    }
    
    /**
     * Caps outliers to specified percentiles
     */
    public Dataset capOutliers(Dataset dataset, String featureName, double lowerPercentile, double upperPercentile) {
        double lowerBound = dataset.calculatePercentile(featureName, lowerPercentile);
        double upperBound = dataset.calculatePercentile(featureName, upperPercentile);
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value < lowerBound) {
                    point.setFeature(featureName, lowerBound);
                } else if (value > upperBound) {
                    point.setFeature(featureName, upperBound);
                }
            }
            return point;
        });
    }
    
    /**
     * Calculates the median absolute deviation
     */
    private double calculateMedianAbsoluteDeviation(Dataset dataset, String featureName) {
        double median = dataset.calculateMedian(featureName);
        
        List<Double> deviations = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) instanceof Number)
                .map(point -> Math.abs(point.getNumericalFeature(featureName) - median))
                .sorted()
                .collect(Collectors.toList());
        
        if (deviations.isEmpty()) {
            return 0.0;
        }
        
        int size = deviations.size();
        if (size % 2 == 0) {
            return (deviations.get(size / 2 - 1) + deviations.get(size / 2)) / 2.0;
        } else {
            return deviations.get(size / 2);
        }
    }
    
    /**
     * Calculates Local Outlier Factor (simplified)
     */
    private double calculateLOF(DataPoint point, List<DataPoint> dataPoints, String featureName, int k) {
        // Simplified LOF calculation
        List<Double> distances = dataPoints.stream()
                .filter(p -> !p.equals(point))
                .map(p -> calculateDistance(point, p, featureName))
                .sorted()
                .collect(Collectors.toList());
        
        if (distances.size() < k) {
            return 1.0;
        }
        
        double kDistance = distances.get(k - 1);
        double reachabilityDensity = k / (distances.subList(0, k).stream().mapToDouble(Double::doubleValue).sum());
        
        // Simplified LOF calculation
        return 1.0 / reachabilityDensity;
    }
    
    /**
     * Calculates distance between two data points for a specific feature
     */
    private double calculateDistance(DataPoint p1, DataPoint p2, String featureName) {
        if (p1.getFeature(featureName) instanceof Number && p2.getFeature(featureName) instanceof Number) {
            double val1 = p1.getNumericalFeature(featureName);
            double val2 = p2.getNumericalFeature(featureName);
            return Math.abs(val1 - val2);
        }
        return Double.MAX_VALUE;
    }
    
    /**
     * Generates outlier detection report
     */
    public void generateOutlierReport(Dataset dataset, String featureName) {
        System.out.println("=== Outlier Detection Report for " + featureName + " ===");
        
        // Z-score outliers
        List<DataPoint> zscoreOutliers = detectOutliersZScore(dataset, featureName, 2.0);
        System.out.printf("Z-score outliers (threshold=2.0): %d (%.2f%%)%n", 
                zscoreOutliers.size(), (double) zscoreOutliers.size() / dataset.size() * 100);
        
        // IQR outliers
        List<DataPoint> iqrOutliers = detectOutliersIQR(dataset, featureName, 1.5);
        System.out.printf("IQR outliers (multiplier=1.5): %d (%.2f%%)%n", 
                iqrOutliers.size(), (double) iqrOutliers.size() / dataset.size() * 100);
        
        // Modified Z-score outliers
        List<DataPoint> modifiedZscoreOutliers = detectOutliersModifiedZScore(dataset, featureName, 3.5);
        System.out.printf("Modified Z-score outliers (threshold=3.5): %d (%.2f%%)%n", 
                modifiedZscoreOutliers.size(), (double) modifiedZscoreOutliers.size() / dataset.size() * 100);
        
        // Feature statistics
        System.out.printf("Feature statistics: mean=%.2f, std=%.2f, min=%.2f, max=%.2f%n",
                dataset.calculateMean(featureName),
                dataset.calculateStandardDeviation(featureName),
                dataset.getMin(featureName),
                dataset.getMax(featureName));
    }
}

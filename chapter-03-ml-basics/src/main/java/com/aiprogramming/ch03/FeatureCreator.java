package com.aiprogramming.ch03;

import java.util.List;
import java.util.Map;

/**
 * Provides methods for creating new features through feature engineering.
 */
public class FeatureCreator {
    
    /**
     * Creates polynomial features
     */
    public Dataset createPolynomialFeatures(Dataset dataset, String featureName, int degree) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                for (int i = 2; i <= degree; i++) {
                    String newFeatureName = featureName + "_pow_" + i;
                    point.setFeature(newFeatureName, Math.pow(value, i));
                }
            }
            return point;
        });
    }
    
    /**
     * Creates interaction features between two numerical features
     */
    public Dataset createInteractionFeatures(Dataset dataset, String feature1, String feature2) {
        return dataset.map(point -> {
            if (point.getFeature(feature1) instanceof Number && point.getFeature(feature2) instanceof Number) {
                double value1 = point.getNumericalFeature(feature1);
                double value2 = point.getNumericalFeature(feature2);
                String interactionName = feature1 + "_x_" + feature2;
                point.setFeature(interactionName, value1 * value2);
            }
            return point;
        });
    }
    
    /**
     * Creates ratio features between two numerical features
     */
    public Dataset createRatioFeatures(Dataset dataset, String numerator, String denominator) {
        return dataset.map(point -> {
            if (point.getFeature(numerator) instanceof Number && point.getFeature(denominator) instanceof Number) {
                double num = point.getNumericalFeature(numerator);
                double den = point.getNumericalFeature(denominator);
                if (Math.abs(den) > 1e-10) { // Avoid division by zero
                    String ratioName = numerator + "_div_" + denominator;
                    point.setFeature(ratioName, num / den);
                }
            }
            return point;
        });
    }
    
    /**
     * Creates difference features between two numerical features
     */
    public Dataset createDifferenceFeatures(Dataset dataset, String feature1, String feature2) {
        return dataset.map(point -> {
            if (point.getFeature(feature1) instanceof Number && point.getFeature(feature2) instanceof Number) {
                double value1 = point.getNumericalFeature(feature1);
                double value2 = point.getNumericalFeature(feature2);
                String diffName = feature1 + "_minus_" + feature2;
                point.setFeature(diffName, value1 - value2);
            }
            return point;
        });
    }
    
    /**
     * Creates sum features between two numerical features
     */
    public Dataset createSumFeatures(Dataset dataset, String feature1, String feature2) {
        return dataset.map(point -> {
            if (point.getFeature(feature1) instanceof Number && point.getFeature(feature2) instanceof Number) {
                double value1 = point.getNumericalFeature(feature1);
                double value2 = point.getNumericalFeature(feature2);
                String sumName = feature1 + "_plus_" + feature2;
                point.setFeature(sumName, value1 + value2);
            }
            return point;
        });
    }
    
    /**
     * Creates binned features from numerical features
     */
    public Dataset createBinnedFeatures(Dataset dataset, String featureName, int numBins) {
        double min = dataset.getMin(featureName);
        double max = dataset.getMax(featureName);
        double binSize = (max - min) / numBins;
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                int bin = (int) ((value - min) / binSize);
                if (bin >= numBins) bin = numBins - 1;
                String binName = featureName + "_bin";
                point.setFeature(binName, bin);
            }
            return point;
        });
    }
    
    /**
     * Creates quantile-based binned features
     */
    public Dataset createQuantileBinnedFeatures(Dataset dataset, String featureName, int numQuantiles) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                int quantile = calculateQuantile(dataset, featureName, value, numQuantiles);
                String quantileName = featureName + "_quantile";
                point.setFeature(quantileName, quantile);
            }
            return point;
        });
    }
    
    /**
     * Creates log-transformed features
     */
    public Dataset createLogFeatures(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value > 0) {
                    String logName = featureName + "_log";
                    point.setFeature(logName, Math.log(value));
                }
            }
            return point;
        });
    }
    
    /**
     * Creates square root transformed features
     */
    public Dataset createSqrtFeatures(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value >= 0) {
                    String sqrtName = featureName + "_sqrt";
                    point.setFeature(sqrtName, Math.sqrt(value));
                }
            }
            return point;
        });
    }
    
    /**
     * Creates reciprocal features
     */
    public Dataset createReciprocalFeatures(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (Math.abs(value) > 1e-10) {
                    String reciprocalName = featureName + "_reciprocal";
                    point.setFeature(reciprocalName, 1.0 / value);
                }
            }
            return point;
        });
    }
    
    /**
     * Creates absolute value features
     */
    public Dataset createAbsFeatures(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                String absName = featureName + "_abs";
                point.setFeature(absName, Math.abs(value));
            }
            return point;
        });
    }
    
    /**
     * Creates rolling statistics features (simplified)
     */
    public Dataset createRollingFeatures(Dataset dataset, String featureName, int windowSize) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        for (int i = 0; i < dataPoints.size(); i++) {
            DataPoint point = dataPoints.get(i);
            
            if (i >= windowSize - 1) {
                // Calculate rolling mean
                double sum = 0.0;
                for (int j = i - windowSize + 1; j <= i; j++) {
                    if (dataPoints.get(j).getFeature(featureName) instanceof Number) {
                        sum += dataPoints.get(j).getNumericalFeature(featureName);
                    }
                }
                double rollingMean = sum / windowSize;
                point.setFeature(featureName + "_rolling_mean_" + windowSize, rollingMean);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Creates lag features for time series data
     */
    public Dataset createLagFeatures(Dataset dataset, String featureName, int lag) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        for (int i = lag; i < dataPoints.size(); i++) {
            DataPoint currentPoint = dataPoints.get(i);
            DataPoint lagPoint = dataPoints.get(i - lag);
            
            if (lagPoint.getFeature(featureName) instanceof Number) {
                double lagValue = lagPoint.getNumericalFeature(featureName);
                currentPoint.setFeature(featureName + "_lag_" + lag, lagValue);
            }
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Creates one-hot encoded features from categorical features
     */
    public Dataset createOneHotFeatures(Dataset dataset, String featureName) {
        // Get unique values
        List<Object> uniqueValues = dataset.getDataPoints().stream()
                .map(point -> point.getFeature(featureName))
                .distinct()
                .toList();
        
        return dataset.map(point -> {
            Object value = point.getFeature(featureName);
            for (Object uniqueValue : uniqueValues) {
                String oneHotName = featureName + "_" + uniqueValue.toString();
                boolean isOne = uniqueValue.equals(value);
                point.setFeature(oneHotName, isOne ? 1 : 0);
            }
            return point;
        });
    }
    
    /**
     * Creates target encoding features (simplified)
     */
    public Dataset createTargetEncodingFeatures(Dataset dataset, String featureName, String targetName) {
        // Calculate target means for each category
        Map<Object, Double> targetMeans = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) != null && point.getTarget() instanceof Number)
                .collect(java.util.stream.Collectors.groupingBy(
                        point -> point.getFeature(featureName),
                        java.util.stream.Collectors.averagingDouble(point -> 
                                ((Number) point.getTarget()).doubleValue())
                ));
        
        return dataset.map(point -> {
            Object value = point.getFeature(featureName);
            if (value != null && targetMeans.containsKey(value)) {
                String targetEncodedName = featureName + "_target_encoded";
                point.setFeature(targetEncodedName, targetMeans.get(value));
            }
            return point;
        });
    }
    
    /**
     * Calculates quantile for a value
     */
    private int calculateQuantile(Dataset dataset, String featureName, double value, int numQuantiles) {
        List<Double> sortedValues = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) instanceof Number)
                .map(point -> point.getNumericalFeature(featureName))
                .sorted()
                .toList();
        
        if (sortedValues.isEmpty()) {
            return 0;
        }
        
        int index = 0;
        for (int i = 0; i < sortedValues.size(); i++) {
            if (sortedValues.get(i) >= value) {
                index = i;
                break;
            }
        }
        
        return (index * numQuantiles) / sortedValues.size();
    }
}

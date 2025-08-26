package com.aiprogramming.ch03;

import java.util.List;

/**
 * Provides various feature scaling and normalization methods for machine learning.
 */
public class FeatureScaler {
    
    /**
     * Min-Max scaling to [0, 1] range
     */
    public Dataset minMaxScale(Dataset dataset, String featureName) {
        double min = dataset.getMin(featureName);
        double max = dataset.getMax(featureName);
        double range = max - min;
        
        if (range == 0) {
            // If all values are the same, set to 0.5
            return dataset.map(point -> {
                point.setFeature(featureName, 0.5);
                return point;
            });
        }
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double scaledValue = (value - min) / range;
                point.setFeature(featureName, scaledValue);
            }
            return point;
        });
    }
    
    /**
     * Standardization (Z-score normalization)
     */
    public Dataset standardize(Dataset dataset, String featureName) {
        double mean = dataset.calculateMean(featureName);
        double std = dataset.calculateStandardDeviation(featureName);
        
        if (std == 0) {
            // If standard deviation is 0, set all values to 0
            return dataset.map(point -> {
                point.setFeature(featureName, 0.0);
                return point;
            });
        }
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double standardizedValue = (value - mean) / std;
                point.setFeature(featureName, standardizedValue);
            }
            return point;
        });
    }
    
    /**
     * Robust scaling using median and IQR
     */
    public Dataset robustScale(Dataset dataset, String featureName) {
        double median = dataset.calculateMedian(featureName);
        double q1 = dataset.calculatePercentile(featureName, 25);
        double q3 = dataset.calculatePercentile(featureName, 75);
        double iqr = q3 - q1;
        
        if (iqr == 0) {
            // If IQR is 0, set all values to 0
            return dataset.map(point -> {
                point.setFeature(featureName, 0.0);
                return point;
            });
        }
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double scaledValue = (value - median) / iqr;
                point.setFeature(featureName, scaledValue);
            }
            return point;
        });
    }
    
    /**
     * Max absolute scaling
     */
    public Dataset maxAbsScale(Dataset dataset, String featureName) {
        double maxAbs = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) instanceof Number)
                .mapToDouble(point -> Math.abs(point.getNumericalFeature(featureName)))
                .max()
                .orElse(1.0);
        
        if (maxAbs == 0) {
            return dataset.map(point -> {
                point.setFeature(featureName, 0.0);
                return point;
            });
        }
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double scaledValue = value / maxAbs;
                point.setFeature(featureName, scaledValue);
            }
            return point;
        });
    }
    
    /**
     * Log transformation
     */
    public Dataset logTransform(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value > 0) {
                    double logValue = Math.log(value);
                    point.setFeature(featureName, logValue);
                } else {
                    // Handle non-positive values
                    point.setFeature(featureName, 0.0);
                }
            }
            return point;
        });
    }
    
    /**
     * Square root transformation
     */
    public Dataset sqrtTransform(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value >= 0) {
                    double sqrtValue = Math.sqrt(value);
                    point.setFeature(featureName, sqrtValue);
                } else {
                    // Handle negative values
                    point.setFeature(featureName, 0.0);
                }
            }
            return point;
        });
    }
    
    /**
     * Box-Cox transformation (simplified version)
     */
    public Dataset boxCoxTransform(Dataset dataset, String featureName, double lambda) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value > 0) {
                    double transformedValue;
                    if (Math.abs(lambda) < 1e-10) {
                        transformedValue = Math.log(value);
                    } else {
                        transformedValue = (Math.pow(value, lambda) - 1) / lambda;
                    }
                    point.setFeature(featureName, transformedValue);
                } else {
                    point.setFeature(featureName, 0.0);
                }
            }
            return point;
        });
    }
    
    /**
     * Yeo-Johnson transformation (simplified version)
     */
    public Dataset yeoJohnsonTransform(Dataset dataset, String featureName, double lambda) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double transformedValue;
                
                if (value >= 0) {
                    if (Math.abs(lambda) < 1e-10) {
                        transformedValue = Math.log(value + 1);
                    } else {
                        transformedValue = (Math.pow(value + 1, lambda) - 1) / lambda;
                    }
                } else {
                    if (Math.abs(lambda - 2) < 1e-10) {
                        transformedValue = -Math.log(-value + 1);
                    } else {
                        transformedValue = -(Math.pow(-value + 1, 2 - lambda) - 1) / (2 - lambda);
                    }
                }
                
                point.setFeature(featureName, transformedValue);
            }
            return point;
        });
    }
    
    /**
     * Quantile transformation
     */
    public Dataset quantileTransform(Dataset dataset, String featureName) {
        List<Double> values = dataset.getDataPoints().stream()
                .filter(point -> point.getFeature(featureName) instanceof Number)
                .map(point -> point.getNumericalFeature(featureName))
                .sorted()
                .toList();
        
        if (values.isEmpty()) {
            return dataset;
        }
        
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                double quantile = calculateQuantile(value, values);
                point.setFeature(featureName, quantile);
            }
            return point;
        });
    }
    
    /**
     * Power transformation
     */
    public Dataset powerTransform(Dataset dataset, String featureName, double power) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (value >= 0) {
                    double transformedValue = Math.pow(value, power);
                    point.setFeature(featureName, transformedValue);
                } else {
                    point.setFeature(featureName, 0.0);
                }
            }
            return point;
        });
    }
    
    /**
     * Reciprocal transformation
     */
    public Dataset reciprocalTransform(Dataset dataset, String featureName) {
        return dataset.map(point -> {
            if (point.getFeature(featureName) instanceof Number) {
                double value = point.getNumericalFeature(featureName);
                if (Math.abs(value) > 1e-10) {
                    double transformedValue = 1.0 / value;
                    point.setFeature(featureName, transformedValue);
                } else {
                    point.setFeature(featureName, 0.0);
                }
            }
            return point;
        });
    }
    
    /**
     * Calculates the quantile of a value in a sorted list
     */
    private double calculateQuantile(double value, List<Double> sortedValues) {
        int size = sortedValues.size();
        int index = 0;
        
        for (int i = 0; i < size; i++) {
            if (sortedValues.get(i) >= value) {
                index = i;
                break;
            }
        }
        
        return (double) index / size;
    }
    
    /**
     * Scales multiple features at once
     */
    public Dataset scaleMultipleFeatures(Dataset dataset, List<String> featureNames, String method) {
        Dataset scaledDataset = dataset.copy();
        
        for (String featureName : featureNames) {
            switch (method.toLowerCase()) {
                case "minmax":
                    scaledDataset = minMaxScale(scaledDataset, featureName);
                    break;
                case "standard":
                    scaledDataset = standardize(scaledDataset, featureName);
                    break;
                case "robust":
                    scaledDataset = robustScale(scaledDataset, featureName);
                    break;
                case "maxabs":
                    scaledDataset = maxAbsScale(scaledDataset, featureName);
                    break;
                default:
                    throw new IllegalArgumentException("Unknown scaling method: " + method);
            }
        }
        
        return scaledDataset;
    }
}

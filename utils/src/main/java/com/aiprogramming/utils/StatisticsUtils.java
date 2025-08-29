package com.aiprogramming.utils;

import java.util.Arrays;
import java.util.Random;

/**
 * Utility class for statistical operations commonly used in AI/ML applications.
 * Provides methods for descriptive statistics, probability distributions, and statistical tests.
 */
public class StatisticsUtils {
    
    /**
     * Calculates the mean (average) of a vector.
     * 
     * @param data Input vector
     * @return Mean value
     */
    public static double mean(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double sum = 0;
        for (double value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    /**
     * Calculates the median of a vector.
     * 
     * @param data Input vector
     * @return Median value
     */
    public static double median(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    /**
     * Calculates the mode (most frequent value) of a vector.
     * 
     * @param data Input vector
     * @return Mode value
     */
    public static double mode(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        // Count frequencies
        java.util.Map<Double, Integer> frequency = new java.util.HashMap<>();
        for (double value : data) {
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }
        
        // Find the most frequent value
        double mode = data[0];
        int maxCount = 1;
        
        for (java.util.Map.Entry<Double, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mode = entry.getKey();
            }
        }
        
        return mode;
    }
    
    /**
     * Calculates the variance of a vector.
     * 
     * @param data Input vector
     * @return Variance
     */
    public static double variance(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double mean = mean(data);
        double sumSquared = 0;
        
        for (double value : data) {
            sumSquared += Math.pow(value - mean, 2);
        }
        
        return sumSquared / data.length;
    }
    
    /**
     * Calculates the sample variance (unbiased estimator).
     * 
     * @param data Input vector
     * @return Sample variance
     */
    public static double sampleVariance(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        if (data.length <= 1) {
            return 0.0;
        }
        
        double mean = mean(data);
        double sumSquared = 0;
        
        for (double value : data) {
            sumSquared += Math.pow(value - mean, 2);
        }
        
        return sumSquared / (data.length - 1);
    }
    
    /**
     * Calculates the standard deviation of a vector.
     * 
     * @param data Input vector
     * @return Standard deviation
     */
    public static double standardDeviation(double[] data) {
        return Math.sqrt(variance(data));
    }
    
    /**
     * Calculates the sample standard deviation.
     * 
     * @param data Input vector
     * @return Sample standard deviation
     */
    public static double sampleStandardDeviation(double[] data) {
        return Math.sqrt(sampleVariance(data));
    }
    
    /**
     * Calculates the coefficient of variation (CV = std/mean).
     * 
     * @param data Input vector
     * @return Coefficient of variation
     */
    public static double coefficientOfVariation(double[] data) {
        double mean = mean(data);
        if (mean == 0) {
            return 0.0;
        }
        return standardDeviation(data) / Math.abs(mean);
    }
    
    /**
     * Calculates the skewness of a vector.
     * 
     * @param data Input vector
     * @return Skewness
     */
    public static double skewness(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double mean = mean(data);
        double std = standardDeviation(data);
        
        if (std == 0) {
            return 0.0;
        }
        
        double sum = 0;
        for (double value : data) {
            sum += Math.pow((value - mean) / std, 3);
        }
        
        return sum / data.length;
    }
    
    /**
     * Calculates the kurtosis of a vector.
     * 
     * @param data Input vector
     * @return Kurtosis
     */
    public static double kurtosis(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double mean = mean(data);
        double std = standardDeviation(data);
        
        if (std == 0) {
            return 0.0;
        }
        
        double sum = 0;
        for (double value : data) {
            sum += Math.pow((value - mean) / std, 4);
        }
        
        return (sum / data.length) - 3; // Excess kurtosis
    }
    
    /**
     * Calculates the correlation coefficient between two vectors.
     * 
     * @param x First vector
     * @param y Second vector
     * @return Correlation coefficient (-1 to 1)
     */
    public static double correlation(double[] x, double[] y) {
        ValidationUtils.validateVector(x, "x");
        ValidationUtils.validateVector(y, "y");
        ValidationUtils.validateVectorLength(y, x.length, "y");
        
        double meanX = mean(x);
        double meanY = mean(y);
        
        double numerator = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;
        
        for (int i = 0; i < x.length; i++) {
            double diffX = x[i] - meanX;
            double diffY = y[i] - meanY;
            
            numerator += diffX * diffY;
            sumXSquared += diffX * diffX;
            sumYSquared += diffY * diffY;
        }
        
        double denominator = Math.sqrt(sumXSquared * sumYSquared);
        if (denominator == 0) {
            return 0.0;
        }
        
        return numerator / denominator;
    }
    
    /**
     * Calculates the covariance between two vectors.
     * 
     * @param x First vector
     * @param y Second vector
     * @return Covariance
     */
    public static double covariance(double[] x, double[] y) {
        ValidationUtils.validateVector(x, "x");
        ValidationUtils.validateVector(y, "y");
        ValidationUtils.validateVectorLength(y, x.length, "y");
        
        double meanX = mean(x);
        double meanY = mean(y);
        
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += (x[i] - meanX) * (y[i] - meanY);
        }
        
        return sum / x.length;
    }
    
    /**
     * Calculates the covariance matrix of a data matrix.
     * 
     * @param data Input data matrix (rows = samples, cols = features)
     * @return Covariance matrix
     */
    public static double[][] covarianceMatrix(double[][] data) {
        ValidationUtils.validateMatrix(data, "data");
        
        int n = data.length;
        int p = data[0].length;
        double[][] covMatrix = new double[p][p];
        
        // Calculate means for each feature
        double[] means = new double[p];
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < n; i++) {
                means[j] += data[i][j];
            }
            means[j] /= n;
        }
        
        // Calculate covariance matrix
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += (data[k][i] - means[i]) * (data[k][j] - means[j]);
                }
                covMatrix[i][j] = sum / n;
            }
        }
        
        return covMatrix;
    }
    
    /**
     * Calculates percentiles of a vector.
     * 
     * @param data Input vector
     * @param percentiles Array of percentiles (0-100)
     * @return Array of percentile values
     */
    public static double[] percentiles(double[] data, double[] percentiles) {
        ValidationUtils.validateVector(data, "data");
        ValidationUtils.validateNotNull(percentiles, "percentiles");
        
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        
        double[] results = new double[percentiles.length];
        for (int i = 0; i < percentiles.length; i++) {
            ValidationUtils.validateRange(percentiles[i], 0, 100, "percentile " + i);
            results[i] = percentile(sorted, percentiles[i]);
        }
        
        return results;
    }
    
    /**
     * Calculates a single percentile.
     * 
     * @param sortedData Sorted data vector
     * @param percentile Percentile (0-100)
     * @return Percentile value
     */
    private static double percentile(double[] sortedData, double percentile) {
        if (percentile <= 0) {
            return sortedData[0];
        }
        if (percentile >= 100) {
            return sortedData[sortedData.length - 1];
        }
        
        double index = (percentile / 100.0) * (sortedData.length - 1);
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);
        
        if (lowerIndex == upperIndex) {
            return sortedData[lowerIndex];
        }
        
        double weight = index - lowerIndex;
        return sortedData[lowerIndex] * (1 - weight) + sortedData[upperIndex] * weight;
    }
    
    /**
     * Calculates the interquartile range (IQR).
     * 
     * @param data Input vector
     * @return IQR (Q3 - Q1)
     */
    public static double interquartileRange(double[] data) {
        double[] quartiles = percentiles(data, new double[]{25, 75});
        return quartiles[1] - quartiles[0];
    }
    
    /**
     * Detects outliers using the IQR method.
     * 
     * @param data Input vector
     * @return Boolean array indicating outliers
     */
    public static boolean[] detectOutliers(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double[] quartiles = percentiles(data, new double[]{25, 75});
        double q1 = quartiles[0];
        double q3 = quartiles[1];
        double iqr = q3 - q1;
        
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;
        
        boolean[] outliers = new boolean[data.length];
        for (int i = 0; i < data.length; i++) {
            outliers[i] = data[i] < lowerBound || data[i] > upperBound;
        }
        
        return outliers;
    }
    
    /**
     * Calculates the z-score for each value in a vector.
     * 
     * @param data Input vector
     * @return Array of z-scores
     */
    public static double[] zScores(double[] data) {
        ValidationUtils.validateVector(data, "data");
        
        double mean = mean(data);
        double std = standardDeviation(data);
        
        if (std == 0) {
            return new double[data.length]; // All z-scores are 0
        }
        
        double[] zScores = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            zScores[i] = (data[i] - mean) / std;
        }
        
        return zScores;
    }
    
    /**
     * Calculates the entropy of a probability distribution.
     * 
     * @param probabilities Array of probabilities (must sum to 1)
     * @return Entropy in bits
     */
    public static double entropy(double[] probabilities) {
        ValidationUtils.validateVector(probabilities, "probabilities");
        
        double sum = 0;
        for (double p : probabilities) {
            ValidationUtils.validateRange(p, 0, 1, "probability");
            if (p > 0) {
                sum -= p * Math.log(p) / Math.log(2);
            }
        }
        
        return sum;
    }
    
    /**
     * Calculates the cross-entropy between two probability distributions.
     * 
     * @param p True probabilities
     * @param q Predicted probabilities
     * @return Cross-entropy
     */
    public static double crossEntropy(double[] p, double[] q) {
        ValidationUtils.validateVector(p, "p");
        ValidationUtils.validateVector(q, "q");
        ValidationUtils.validateVectorLength(q, p.length, "q");
        
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            ValidationUtils.validateRange(p[i], 0, 1, "p[" + i + "]");
            ValidationUtils.validateRange(q[i], 0, 1, "q[" + i + "]");
            
            if (p[i] > 0 && q[i] > 0) {
                sum -= p[i] * Math.log(q[i]) / Math.log(2);
            }
        }
        
        return sum;
    }
    
    /**
     * Calculates the KL divergence between two probability distributions.
     * 
     * @param p True probabilities
     * @param q Predicted probabilities
     * @return KL divergence
     */
    public static double klDivergence(double[] p, double[] q) {
        ValidationUtils.validateVector(p, "p");
        ValidationUtils.validateVector(q, "q");
        ValidationUtils.validateVectorLength(q, p.length, "q");
        
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            ValidationUtils.validateRange(p[i], 0, 1, "p[" + i + "]");
            ValidationUtils.validateRange(q[i], 0, 1, "q[" + i + "]");
            
            if (p[i] > 0 && q[i] > 0) {
                sum += p[i] * Math.log(p[i] / q[i]) / Math.log(2);
            }
        }
        
        return sum;
    }
    
    /**
     * Generates random numbers from a normal distribution.
     * 
     * @param n Number of samples
     * @param mean Mean of the distribution
     * @param std Standard deviation of the distribution
     * @param seed Random seed
     * @return Array of random numbers
     */
    public static double[] randomNormal(int n, double mean, double std, long seed) {
        ValidationUtils.validatePositive(n, "n");
        ValidationUtils.validateNonNegative(std, "std");
        
        Random random = new Random(seed);
        double[] samples = new double[n];
        
        for (int i = 0; i < n; i++) {
            samples[i] = mean + std * random.nextGaussian();
        }
        
        return samples;
    }
    
    /**
     * Generates random numbers from a uniform distribution.
     * 
     * @param n Number of samples
     * @param min Minimum value
     * @param max Maximum value
     * @param seed Random seed
     * @return Array of random numbers
     */
    public static double[] randomUniform(int n, double min, double max, long seed) {
        ValidationUtils.validatePositive(n, "n");
        
        Random random = new Random(seed);
        double[] samples = new double[n];
        
        for (int i = 0; i < n; i++) {
            samples[i] = min + (max - min) * random.nextDouble();
        }
        
        return samples;
    }
}

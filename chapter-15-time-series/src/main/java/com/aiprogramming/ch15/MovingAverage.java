package com.aiprogramming.ch15;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Implements various moving average smoothing techniques for time series
 */
public class MovingAverage {
    
    /**
     * Simple Moving Average (SMA)
     * Calculates the average of the last n values
     */
    public static double[] simpleMovingAverage(double[] data, int windowSize) {
        if (windowSize <= 0 || windowSize > data.length) {
            throw new IllegalArgumentException("Invalid window size: " + windowSize);
        }
        
        double[] result = new double[data.length];
        
        // Fill the beginning with NaN or original values
        for (int i = 0; i < windowSize - 1; i++) {
            result[i] = Double.NaN;
        }
        
        // Calculate SMA for the rest
        for (int i = windowSize - 1; i < data.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < windowSize; j++) {
                sum += data[i - j];
            }
            result[i] = sum / windowSize;
        }
        
        return result;
    }
    
    /**
     * Weighted Moving Average (WMA)
     * Gives more weight to recent values
     */
    public static double[] weightedMovingAverage(double[] data, int windowSize) {
        if (windowSize <= 0 || windowSize > data.length) {
            throw new IllegalArgumentException("Invalid window size: " + windowSize);
        }
        
        double[] result = new double[data.length];
        
        // Fill the beginning with NaN
        for (int i = 0; i < windowSize - 1; i++) {
            result[i] = Double.NaN;
        }
        
        // Calculate WMA for the rest
        double weightSum = windowSize * (windowSize + 1) / 2.0; // Sum of weights
        
        for (int i = windowSize - 1; i < data.length; i++) {
            double weightedSum = 0.0;
            for (int j = 0; j < windowSize; j++) {
                weightedSum += (j + 1) * data[i - j];
            }
            result[i] = weightedSum / weightSum;
        }
        
        return result;
    }
    
    /**
     * Exponential Moving Average (EMA)
     * Gives exponentially decreasing weight to older values
     */
    public static double[] exponentialMovingAverage(double[] data, double alpha) {
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1: " + alpha);
        }
        
        double[] result = new double[data.length];
        
        // Initialize with the first value
        result[0] = data[0];
        
        // Calculate EMA for the rest
        for (int i = 1; i < data.length; i++) {
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
        }
        
        return result;
    }
    
    /**
     * Double Exponential Moving Average (DEMA)
     * Reduces lag by applying EMA twice
     */
    public static double[] doubleExponentialMovingAverage(double[] data, double alpha) {
        double[] ema1 = exponentialMovingAverage(data, alpha);
        double[] ema2 = exponentialMovingAverage(ema1, alpha);
        
        double[] dema = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            dema[i] = 2 * ema1[i] - ema2[i];
        }
        
        return dema;
    }
    
    /**
     * Triple Exponential Moving Average (TEMA)
     * Further reduces lag by applying EMA three times
     */
    public static double[] tripleExponentialMovingAverage(double[] data, double alpha) {
        double[] ema1 = exponentialMovingAverage(data, alpha);
        double[] ema2 = exponentialMovingAverage(ema1, alpha);
        double[] ema3 = exponentialMovingAverage(ema2, alpha);
        
        double[] tema = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            tema[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i];
        }
        
        return tema;
    }
    
    /**
     * Adaptive Moving Average (AMA)
     * Adjusts smoothing based on market volatility
     */
    public static double[] adaptiveMovingAverage(double[] data, int fastPeriod, int slowPeriod) {
        if (fastPeriod >= slowPeriod) {
            throw new IllegalArgumentException("Fast period must be less than slow period");
        }
        
        double[] result = new double[data.length];
        result[0] = data[0];
        
        for (int i = 1; i < data.length; i++) {
            // Calculate efficiency ratio
            double change = Math.abs(data[i] - data[i - 1]);
            double volatility = 0.0;
            
            for (int j = Math.max(0, i - slowPeriod); j < i; j++) {
                volatility += Math.abs(data[j + 1] - data[j]);
            }
            
            double efficiencyRatio = volatility > 0 ? change / volatility : 0.0;
            
            // Calculate smoothing constant
            double fastSC = 2.0 / (fastPeriod + 1);
            double slowSC = 2.0 / (slowPeriod + 1);
            double amaSC = Math.pow(efficiencyRatio * (fastSC - slowSC) + slowSC, 2);
            
            // Apply AMA
            result[i] = amaSC * data[i] + (1 - amaSC) * result[i - 1];
        }
        
        return result;
    }
    
    /**
     * Calculate the optimal alpha for EMA based on data characteristics
     */
    public static double calculateOptimalAlpha(double[] data) {
        DescriptiveStatistics stats = new DescriptiveStatistics(data);
        double stdDev = stats.getStandardDeviation();
        double mean = stats.getMean();
        
        // Simple heuristic: higher volatility -> higher alpha
        double coefficientOfVariation = stdDev / mean;
        return Math.min(0.5, Math.max(0.1, coefficientOfVariation));
    }
}

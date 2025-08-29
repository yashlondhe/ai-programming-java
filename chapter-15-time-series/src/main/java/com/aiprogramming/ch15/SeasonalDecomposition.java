package com.aiprogramming.ch15;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Implements seasonal decomposition of time series into trend, seasonal, and residual components
 */
public class SeasonalDecomposition {
    
    /**
     * Result of seasonal decomposition
     */
    public static class DecompositionResult {
        private double[] trend;
        private double[] seasonal;
        private double[] residual;
        private double[] original;
        
        public DecompositionResult(double[] original, double[] trend, double[] seasonal, double[] residual) {
            this.original = original;
            this.trend = trend;
            this.seasonal = seasonal;
            this.residual = residual;
        }
        
        public double[] getTrend() { return trend; }
        public double[] getSeasonal() { return seasonal; }
        public double[] getResidual() { return residual; }
        public double[] getOriginal() { return original; }
    }
    
    /**
     * Classical decomposition (additive model)
     * Y(t) = Trend(t) + Seasonal(t) + Residual(t)
     */
    public static DecompositionResult classicalDecomposition(double[] data, int period) {
        if (period <= 1 || period >= data.length / 2) {
            throw new IllegalArgumentException("Invalid period: " + period);
        }
        
        // Step 1: Calculate trend using moving average
        double[] trend = calculateTrend(data, period);
        
        // Step 2: Detrend the data
        double[] detrended = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            detrended[i] = data[i] - trend[i];
        }
        
        // Step 3: Calculate seasonal component
        double[] seasonal = calculateSeasonal(detrended, period);
        
        // Step 4: Calculate residuals
        double[] residual = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            residual[i] = detrended[i] - seasonal[i];
        }
        
        return new DecompositionResult(data, trend, seasonal, residual);
    }
    
    /**
     * Multiplicative decomposition
     * Y(t) = Trend(t) * Seasonal(t) * Residual(t)
     */
    public static DecompositionResult multiplicativeDecomposition(double[] data, int period) {
        if (period <= 1 || period >= data.length / 2) {
            throw new IllegalArgumentException("Invalid period: " + period);
        }
        
        // Step 1: Calculate trend using moving average
        double[] trend = calculateTrend(data, period);
        
        // Step 2: Detrend the data (division instead of subtraction)
        double[] detrended = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            detrended[i] = trend[i] != 0 ? data[i] / trend[i] : 1.0;
        }
        
        // Step 3: Calculate seasonal component
        double[] seasonal = calculateSeasonal(detrended, period);
        
        // Step 4: Calculate residuals
        double[] residual = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            residual[i] = seasonal[i] != 0 ? detrended[i] / seasonal[i] : 1.0;
        }
        
        return new DecompositionResult(data, trend, seasonal, residual);
    }
    
    /**
     * Calculate trend using centered moving average
     */
    private static double[] calculateTrend(double[] data, int period) {
        double[] trend = new double[data.length];
        
        // Use centered moving average for trend
        int windowSize = period % 2 == 0 ? period : period + 1;
        int halfWindow = windowSize / 2;
        
        // Fill beginning and end with NaN
        for (int i = 0; i < halfWindow; i++) {
            trend[i] = Double.NaN;
            trend[data.length - 1 - i] = Double.NaN;
        }
        
        // Calculate centered moving average
        for (int i = halfWindow; i < data.length - halfWindow; i++) {
            double sum = 0.0;
            for (int j = -halfWindow; j <= halfWindow; j++) {
                sum += data[i + j];
            }
            trend[i] = sum / windowSize;
        }
        
        return trend;
    }
    
    /**
     * Calculate seasonal component
     */
    private static double[] calculateSeasonal(double[] detrended, int period) {
        double[] seasonal = new double[detrended.length];
        
        // Calculate seasonal indices for each position in the period
        double[] seasonalIndices = new double[period];
        int[] counts = new int[period];
        
        for (int i = 0; i < detrended.length; i++) {
            if (!Double.isNaN(detrended[i])) {
                int pos = i % period;
                seasonalIndices[pos] += detrended[i];
                counts[pos]++;
            }
        }
        
        // Average the seasonal indices
        for (int i = 0; i < period; i++) {
            if (counts[i] > 0) {
                seasonalIndices[i] /= counts[i];
            }
        }
        
        // Center the seasonal indices (mean should be 0 for additive model)
        double mean = 0.0;
        for (double index : seasonalIndices) {
            mean += index;
        }
        mean /= period;
        
        for (int i = 0; i < period; i++) {
            seasonalIndices[i] -= mean;
        }
        
        // Apply seasonal pattern to all data points
        for (int i = 0; i < detrended.length; i++) {
            seasonal[i] = seasonalIndices[i % period];
        }
        
        return seasonal;
    }
    
    /**
     * X-13ARIMA-SEATS decomposition (simplified version)
     * Uses more sophisticated filtering and seasonal adjustment
     */
    public static DecompositionResult x13Decomposition(double[] data, int period) {
        // This is a simplified version of X-13ARIMA-SEATS
        // In practice, you would use specialized libraries like RJDemetra
        
        // For now, we'll use a more robust trend calculation
        double[] trend = robustTrend(data, period);
        
        // Detrend and calculate seasonal
        double[] detrended = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            detrended[i] = data[i] - trend[i];
        }
        
        double[] seasonal = robustSeasonal(detrended, period);
        
        // Calculate residuals
        double[] residual = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            residual[i] = detrended[i] - seasonal[i];
        }
        
        return new DecompositionResult(data, trend, seasonal, residual);
    }
    
    /**
     * Robust trend calculation using median-based smoothing
     */
    private static double[] robustTrend(double[] data, int period) {
        double[] trend = new double[data.length];
        int windowSize = Math.max(period, 7); // Minimum window size
        
        for (int i = 0; i < data.length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(data.length - 1, i + windowSize / 2);
            
            // Use median for robust trend estimation
            DescriptiveStatistics stats = new DescriptiveStatistics();
            for (int j = start; j <= end; j++) {
                stats.addValue(data[j]);
            }
            trend[i] = stats.getPercentile(50); // Median
        }
        
        return trend;
    }
    
    /**
     * Robust seasonal calculation using median-based seasonal indices
     */
    private static double[] robustSeasonal(double[] detrended, int period) {
        double[] seasonal = new double[detrended.length];
        
        // Group values by position in period
        double[][] groups = new double[period][];
        int[] groupSizes = new int[period];
        
        // Count group sizes
        for (int i = 0; i < detrended.length; i++) {
            if (!Double.isNaN(detrended[i])) {
                groupSizes[i % period]++;
            }
        }
        
        // Initialize groups
        for (int i = 0; i < period; i++) {
            groups[i] = new double[groupSizes[i]];
        }
        
        // Fill groups
        int[] groupIndices = new int[period];
        for (int i = 0; i < detrended.length; i++) {
            if (!Double.isNaN(detrended[i])) {
                int pos = i % period;
                groups[pos][groupIndices[pos]++] = detrended[i];
            }
        }
        
        // Calculate median for each group
        double[] seasonalIndices = new double[period];
        for (int i = 0; i < period; i++) {
            DescriptiveStatistics stats = new DescriptiveStatistics(groups[i]);
            seasonalIndices[i] = stats.getPercentile(50);
        }
        
        // Center the seasonal indices
        double mean = 0.0;
        for (double index : seasonalIndices) {
            mean += index;
        }
        mean /= period;
        
        for (int i = 0; i < period; i++) {
            seasonalIndices[i] -= mean;
        }
        
        // Apply seasonal pattern
        for (int i = 0; i < detrended.length; i++) {
            seasonal[i] = seasonalIndices[i % period];
        }
        
        return seasonal;
    }
}

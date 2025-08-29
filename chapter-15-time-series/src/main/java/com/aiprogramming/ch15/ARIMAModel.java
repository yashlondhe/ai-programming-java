package com.aiprogramming.ch15;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

/**
 * Implements ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting
 */
public class ARIMAModel {
    private int p; // AR order
    private int d; // Differencing order
    private int q; // MA order
    private double[] arCoefficients;
    private double[] maCoefficients;
    private double[] residuals;
    private double[] originalData;
    private double[] differencedData;
    
    public ARIMAModel(int p, int d, int q) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.arCoefficients = new double[p];
        this.maCoefficients = new double[q];
    }
    
    /**
     * Fit the ARIMA model to the data
     */
    public void fit(double[] data) {
        this.originalData = data.clone();
        
        // Step 1: Apply differencing
        this.differencedData = applyDifferencing(data, d);
        
        // Step 2: Estimate AR coefficients
        if (p > 0) {
            this.arCoefficients = estimateARCoefficients(differencedData, p);
        }
        
        // Step 3: Estimate MA coefficients
        if (q > 0) {
            this.maCoefficients = estimateMACoefficients(differencedData, q);
        }
        
        // Step 4: Calculate residuals
        this.residuals = calculateResiduals();
    }
    
    /**
     * Apply differencing to make the series stationary
     */
    private double[] applyDifferencing(double[] data, int order) {
        double[] result = data.clone();
        
        for (int i = 0; i < order; i++) {
            double[] temp = new double[result.length - 1];
            for (int j = 0; j < temp.length; j++) {
                temp[j] = result[j + 1] - result[j];
            }
            result = temp;
        }
        
        return result;
    }
    
    /**
     * Apply inverse differencing to restore the original scale
     */
    private double[] applyInverseDifferencing(double[] data, int order) {
        double[] result = data.clone();
        
        for (int i = 0; i < order; i++) {
            double[] temp = new double[result.length + 1];
            temp[0] = originalData[originalData.length - result.length - 1 + i];
            
            for (int j = 1; j < temp.length; j++) {
                temp[j] = temp[j - 1] + result[j - 1];
            }
            result = temp;
        }
        
        return result;
    }
    
    /**
     * Estimate AR coefficients using Yule-Walker equations
     */
    private double[] estimateARCoefficients(double[] data, int order) {
        if (order == 0) return new double[0];
        
        // Calculate autocorrelations
        double[] autocorr = calculateAutocorrelations(data, order);
        
        // Build Toeplitz matrix
        RealMatrix toeplitz = new Array2DRowRealMatrix(order, order);
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                toeplitz.setEntry(i, j, autocorr[Math.abs(i - j)]);
            }
        }
        
        // Build right-hand side vector
        RealVector rhs = new ArrayRealVector(order);
        for (int i = 0; i < order; i++) {
            rhs.setEntry(i, autocorr[i + 1]);
        }
        
        // Solve Yule-Walker equations
        DecompositionSolver solver = new LUDecomposition(toeplitz).getSolver();
        RealVector solution = solver.solve(rhs);
        
        double[] coefficients = new double[order];
        for (int i = 0; i < order; i++) {
            coefficients[i] = solution.getEntry(i);
        }
        
        return coefficients;
    }
    
    /**
     * Estimate MA coefficients using innovation algorithm
     */
    private double[] estimateMACoefficients(double[] data, int order) {
        if (order == 0) return new double[0];
        
        // Initialize with zeros
        double[] coefficients = new double[order];
        
        // Simple estimation using autocorrelations
        double[] autocorr = calculateAutocorrelations(data, order);
        
        // Use first-order approximation
        if (order >= 1) {
            coefficients[0] = autocorr[1] / autocorr[0];
        }
        
        // For higher orders, use iterative approach
        for (int i = 1; i < order; i++) {
            coefficients[i] = autocorr[i + 1] / autocorr[0];
        }
        
        return coefficients;
    }
    
    /**
     * Calculate autocorrelations
     */
    private double[] calculateAutocorrelations(double[] data, int maxLag) {
        double[] autocorr = new double[maxLag + 1];
        double mean = 0.0;
        
        // Calculate mean
        for (double value : data) {
            mean += value;
        }
        mean /= data.length;
        
        // Calculate variance
        double variance = 0.0;
        for (double value : data) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= data.length;
        
        // Calculate autocorrelations
        for (int lag = 0; lag <= maxLag; lag++) {
            double sum = 0.0;
            for (int i = 0; i < data.length - lag; i++) {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            autocorr[lag] = sum / ((data.length - lag) * variance);
        }
        
        return autocorr;
    }
    
    /**
     * Calculate residuals
     */
    private double[] calculateResiduals() {
        double[] residuals = new double[differencedData.length];
        
        for (int i = 0; i < differencedData.length; i++) {
            double prediction = 0.0;
            
            // AR component
            for (int j = 0; j < p && i - j - 1 >= 0; j++) {
                prediction += arCoefficients[j] * differencedData[i - j - 1];
            }
            
            // MA component
            for (int j = 0; j < q && i - j - 1 >= 0; j++) {
                prediction += maCoefficients[j] * residuals[i - j - 1];
            }
            
            residuals[i] = differencedData[i] - prediction;
        }
        
        return residuals;
    }
    
    /**
     * Forecast future values
     */
    public double[] forecast(int steps) {
        if (differencedData == null) {
            throw new IllegalStateException("Model must be fitted before forecasting");
        }
        
        double[] forecast = new double[steps];
        double[] lastValues = new double[Math.max(p, q)];
        
        // Initialize with last known values
        for (int i = 0; i < lastValues.length; i++) {
            int idx = differencedData.length - lastValues.length + i;
            if (idx >= 0) {
                lastValues[i] = differencedData[idx];
            }
        }
        
        double[] lastResiduals = new double[q];
        for (int i = 0; i < q; i++) {
            int idx = residuals.length - q + i;
            if (idx >= 0) {
                lastResiduals[i] = residuals[idx];
            }
        }
        
        // Generate forecasts
        for (int t = 0; t < steps; t++) {
            double prediction = 0.0;
            
            // AR component
            for (int i = 0; i < p; i++) {
                prediction += arCoefficients[i] * lastValues[lastValues.length - 1 - i];
            }
            
            // MA component
            for (int i = 0; i < q; i++) {
                prediction += maCoefficients[i] * lastResiduals[lastResiduals.length - 1 - i];
            }
            
            forecast[t] = prediction;
            
            // Update last values
            for (int i = 0; i < lastValues.length - 1; i++) {
                lastValues[i] = lastValues[i + 1];
            }
            lastValues[lastValues.length - 1] = prediction;
            
            // Update residuals (assume zero for future)
            for (int i = 0; i < lastResiduals.length - 1; i++) {
                lastResiduals[i] = lastResiduals[i + 1];
            }
            lastResiduals[lastResiduals.length - 1] = 0.0;
        }
        
        // Apply inverse differencing
        return applyInverseDifferencing(forecast, d);
    }
    
    /**
     * Calculate AIC (Akaike Information Criterion)
     */
    public double calculateAIC() {
        if (residuals == null) {
            throw new IllegalStateException("Model must be fitted before calculating AIC");
        }
        
        int n = residuals.length;
        int k = p + q + 1; // Number of parameters + variance
        
        // Calculate residual sum of squares
        double rss = 0.0;
        for (double residual : residuals) {
            rss += residual * residual;
        }
        
        return n * Math.log(rss / n) + 2 * k;
    }
    
    /**
     * Calculate BIC (Bayesian Information Criterion)
     */
    public double calculateBIC() {
        if (residuals == null) {
            throw new IllegalStateException("Model must be fitted before calculating BIC");
        }
        
        int n = residuals.length;
        int k = p + q + 1; // Number of parameters + variance
        
        // Calculate residual sum of squares
        double rss = 0.0;
        for (double residual : residuals) {
            rss += residual * residual;
        }
        
        return n * Math.log(rss / n) + k * Math.log(n);
    }
    
    /**
     * Get model parameters
     */
    public int getP() { return p; }
    public int getD() { return d; }
    public int getQ() { return q; }
    public double[] getARCoefficients() { return arCoefficients.clone(); }
    public double[] getMACoefficients() { return maCoefficients.clone(); }
    
    @Override
    public String toString() {
        return String.format("ARIMA(%d,%d,%d) - AIC: %.2f, BIC: %.2f", 
                           p, d, q, calculateAIC(), calculateBIC());
    }
}

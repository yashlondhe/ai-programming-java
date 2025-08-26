package com.aiprogramming.ch05;

import java.util.*;

/**
 * Container class for regression evaluation results
 */
public class RegressionMetrics {
    private final double mae;
    private final double mse;
    private final double rmse;
    private final double r2;
    private final double mape;
    private final List<Double> residuals;
    
    public RegressionMetrics(double mae, double mse, double rmse, double r2, 
                           double mape, List<Double> residuals) {
        this.mae = mae;
        this.mse = mse;
        this.rmse = rmse;
        this.r2 = r2;
        this.mape = mape;
        this.residuals = new ArrayList<>(residuals);
    }
    
    // Getters
    public double getMAE() { return mae; }
    public double getMSE() { return mse; }
    public double getRMSE() { return rmse; }
    public double getR2() { return r2; }
    public double getMAPE() { return mape; }
    public List<Double> getResiduals() { return new ArrayList<>(residuals); }
    
    public void printMetrics() {
        System.out.printf("Mean Absolute Error (MAE): %.4f%n", mae);
        System.out.printf("Mean Squared Error (MSE): %.4f%n", mse);
        System.out.printf("Root Mean Squared Error (RMSE): %.4f%n", rmse);
        System.out.printf("R² Score: %.4f%n", r2);
        System.out.printf("Mean Absolute Percentage Error (MAPE): %.2f%%%n", mape);
        
        // Basic residual statistics
        double meanResidual = residuals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdResidual = Math.sqrt(residuals.stream()
                .mapToDouble(r -> Math.pow(r - meanResidual, 2))
                .average()
                .orElse(0.0));
        
        System.out.printf("Mean Residual: %.4f%n", meanResidual);
        System.out.printf("Std Residual: %.4f%n", stdResidual);
    }
}
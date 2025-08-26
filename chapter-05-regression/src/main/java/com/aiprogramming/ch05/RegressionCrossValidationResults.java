package com.aiprogramming.ch05;

import java.util.*;

/**
 * Cross-validation results container for regression
 */
public class RegressionCrossValidationResults {
    private final List<Double> maes;
    private final List<Double> mses;
    private final List<Double> rmses;
    private final List<Double> r2s;
    
    public RegressionCrossValidationResults(List<Double> maes, List<Double> mses, 
                                          List<Double> rmses, List<Double> r2s) {
        this.maes = maes;
        this.mses = mses;
        this.rmses = rmses;
        this.r2s = r2s;
    }
    
    public double getMeanMAE() {
        return maes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdMAE() {
        double mean = getMeanMAE();
        double variance = maes.stream()
                .mapToDouble(mae -> Math.pow(mae - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public double getMeanR2() {
        return r2s.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdR2() {
        double mean = getMeanR2();
        double variance = r2s.stream()
                .mapToDouble(r2 -> Math.pow(r2 - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public double getMeanMSE() {
        return mses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdMSE() {
        double mean = getMeanMSE();
        double variance = mses.stream()
                .mapToDouble(mse -> Math.pow(mse - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public double getMeanRMSE() {
        return rmses.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getStdRMSE() {
        double mean = getMeanRMSE();
        double variance = rmses.stream()
                .mapToDouble(rmse -> Math.pow(rmse - mean, 2))
                .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    public void printResults() {
        System.out.printf("Cross-Validation Results (Mean ± Std):%n");
        System.out.printf("MAE: %.4f ± %.4f%n", getMeanMAE(), getStdMAE());
        System.out.printf("MSE: %.4f ± %.4f%n", getMeanMSE(), getStdMSE());
        System.out.printf("RMSE: %.4f ± %.4f%n", getMeanRMSE(), getStdRMSE());
        System.out.printf("R²: %.4f ± %.4f%n", getMeanR2(), getStdR2());
    }
}
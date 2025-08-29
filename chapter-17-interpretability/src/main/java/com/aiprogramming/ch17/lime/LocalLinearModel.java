package com.aiprogramming.ch17.lime;

/**
 * Local linear model used in LIME for approximating model behavior
 * around a specific instance
 */
public class LocalLinearModel {
    
    private final double[] coefficients;
    private final double rSquared;
    
    /**
     * Constructor for local linear model
     * 
     * @param coefficients regression coefficients (including intercept)
     * @param rSquared R-squared value indicating model fit quality
     */
    public LocalLinearModel(double[] coefficients, double rSquared) {
        this.coefficients = coefficients.clone();
        this.rSquared = rSquared;
    }
    
    /**
     * Get the regression coefficients
     * 
     * @return array of coefficients (first element is intercept)
     */
    public double[] getCoefficients() {
        return coefficients.clone();
    }
    
    /**
     * Get the intercept term
     * 
     * @return intercept value
     */
    public double getIntercept() {
        return coefficients.length > 0 ? coefficients[0] : 0.0;
    }
    
    /**
     * Get feature coefficients (excluding intercept)
     * 
     * @return array of feature coefficients
     */
    public double[] getFeatureCoefficients() {
        if (coefficients.length <= 1) {
            return new double[0];
        }
        
        double[] featureCoeffs = new double[coefficients.length - 1];
        System.arraycopy(coefficients, 1, featureCoeffs, 0, featureCoeffs.length);
        return featureCoeffs;
    }
    
    /**
     * Get the R-squared value
     * 
     * @return R-squared value
     */
    public double getRSquared() {
        return rSquared;
    }
    
    /**
     * Predict using the local linear model
     * 
     * @param features input features
     * @return predicted value
     */
    public double predict(double[] features) {
        if (coefficients.length == 0) {
            return 0.0;
        }
        
        double prediction = coefficients[0]; // Intercept
        
        for (int i = 0; i < Math.min(features.length, coefficients.length - 1); i++) {
            prediction += coefficients[i + 1] * features[i];
        }
        
        return prediction;
    }
    
    /**
     * Get the number of features in the model
     * 
     * @return number of features
     */
    public int getNumFeatures() {
        return Math.max(0, coefficients.length - 1);
    }
    
    /**
     * Check if the model is well-fitted
     * 
     * @return true if R-squared is above threshold
     */
    public boolean isWellFitted() {
        return rSquared > 0.5; // Threshold can be adjusted
    }
    
    /**
     * Get model quality description
     * 
     * @return string describing model quality
     */
    public String getQualityDescription() {
        if (rSquared >= 0.8) {
            return "Excellent fit";
        } else if (rSquared >= 0.6) {
            return "Good fit";
        } else if (rSquared >= 0.4) {
            return "Fair fit";
        } else {
            return "Poor fit";
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LocalLinearModel{");
        sb.append("intercept=").append(getIntercept());
        sb.append(", rSquared=").append(rSquared);
        sb.append(", quality=").append(getQualityDescription());
        sb.append("}");
        return sb.toString();
    }
}

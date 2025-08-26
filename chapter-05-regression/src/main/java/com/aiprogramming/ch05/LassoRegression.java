package com.aiprogramming.ch05;

import java.util.*;

/**
 * Lasso Regression Implementation with L1 Regularization using Coordinate Descent
 */
public class LassoRegression implements Regressor {
    
    private final double alpha;
    private final int maxIterations;
    private final double tolerance;
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public LassoRegression(double alpha) {
        this(alpha, 1000, 1e-6);
    }
    
    public LassoRegression(double alpha, int maxIterations, double tolerance) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.coefficients = new HashMap<>();
        this.intercept = 0.0;
        this.featureNames = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Convert data to matrix format
        int numSamples = trainingData.size();
        int numFeatures = featureNames.size();
        
        // X matrix and y vector
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        // Standardize features
        double[] featureMeans = calculateFeatureMeans(X);
        double[] featureStds = calculateFeatureStds(X, featureMeans);
        standardizeFeatures(X, featureMeans, featureStds);
        
        // Center target
        double targetMean = Arrays.stream(y).average().orElse(0.0);
        for (int i = 0; i < y.length; i++) {
            y[i] -= targetMean;
        }
        
        // Initialize coefficients
        double[] beta = new double[numFeatures];
        
        // Coordinate descent algorithm
        for (int iter = 0; iter < maxIterations; iter++) {
            double[] oldBeta = Arrays.copyOf(beta, beta.length);
            
            for (int j = 0; j < numFeatures; j++) {
                // Calculate residual without j-th feature
                double[] residual = calculateResidual(X, y, beta, j);
                
                // Calculate correlation with j-th feature
                double correlation = calculateCorrelation(X, residual, j);
                
                // Soft thresholding operator
                beta[j] = softThreshold(correlation, alpha);
            }
            
            // Check convergence
            if (hasConverged(beta, oldBeta)) {
                break;
            }
        }
        
        // Transform coefficients back to original scale
        this.coefficients.clear();
        for (int i = 0; i < numFeatures; i++) {
            double originalCoeff = (featureStds[i] != 0) ? beta[i] / featureStds[i] : 0.0;
            coefficients.put(featureNames.get(i), originalCoeff);
        }
        
        // Calculate intercept
        this.intercept = targetMean;
        for (int i = 0; i < numFeatures; i++) {
            this.intercept -= coefficients.get(featureNames.get(i)) * featureMeans[i];
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double prediction = intercept;
        
        for (String featureName : featureNames) {
            double featureValue = features.getOrDefault(featureName, 0.0);
            double coefficient = coefficients.getOrDefault(featureName, 0.0);
            prediction += coefficient * featureValue;
        }
        
        return prediction;
    }
    
    /**
     * Gets the regularization parameter
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Gets the learned coefficients
     */
    public Map<String, Double> getCoefficients() {
        return new HashMap<>(coefficients);
    }
    
    /**
     * Gets the learned intercept
     */
    public double getIntercept() {
        return intercept;
    }
    
    /**
     * Gets the selected features (non-zero coefficients)
     */
    public Set<String> getSelectedFeatures() {
        Set<String> selectedFeatures = new HashSet<>();
        for (Map.Entry<String, Double> entry : coefficients.entrySet()) {
            if (Math.abs(entry.getValue()) > 1e-10) {
                selectedFeatures.add(entry.getKey());
            }
        }
        return selectedFeatures;
    }
    
    /**
     * Gets the sparsity level (percentage of zero coefficients)
     */
    public double getSparsity() {
        long zeroCoeffs = coefficients.values().stream()
                .mapToLong(coeff -> Math.abs(coeff) <= 1e-10 ? 1 : 0)
                .sum();
        return (double) zeroCoeffs / coefficients.size();
    }
    
    private double[] calculateFeatureMeans(double[][] X) {
        int numFeatures = X[0].length;
        double[] means = new double[numFeatures];
        
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            means[j] = sum / X.length;
        }
        
        return means;
    }
    
    private double[] calculateFeatureStds(double[][] X, double[] means) {
        int numFeatures = X[0].length;
        double[] stds = new double[numFeatures];
        
        for (int j = 0; j < numFeatures; j++) {
            double sumSquares = 0.0;
            for (int i = 0; i < X.length; i++) {
                double diff = X[i][j] - means[j];
                sumSquares += diff * diff;
            }
            stds[j] = Math.sqrt(sumSquares / X.length);
        }
        
        return stds;
    }
    
    private void standardizeFeatures(double[][] X, double[] means, double[] stds) {
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                if (stds[j] != 0) {
                    X[i][j] = (X[i][j] - means[j]) / stds[j];
                } else {
                    X[i][j] = 0.0;
                }
            }
        }
    }
    
    private double[] calculateResidual(double[][] X, double[] y, double[] beta, int excludeFeature) {
        double[] residual = Arrays.copyOf(y, y.length);
        
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < beta.length; j++) {
                if (j != excludeFeature) {
                    residual[i] -= beta[j] * X[i][j];
                }
            }
        }
        
        return residual;
    }
    
    private double calculateCorrelation(double[][] X, double[] residual, int featureIndex) {
        double correlation = 0.0;
        
        for (int i = 0; i < X.length; i++) {
            correlation += X[i][featureIndex] * residual[i];
        }
        
        return correlation / X.length;
    }
    
    private double softThreshold(double value, double threshold) {
        if (value > threshold) {
            return value - threshold;
        } else if (value < -threshold) {
            return value + threshold;
        } else {
            return 0.0;
        }
    }
    
    private boolean hasConverged(double[] newBeta, double[] oldBeta) {
        double maxDiff = 0.0;
        
        for (int i = 0; i < newBeta.length; i++) {
            double diff = Math.abs(newBeta[i] - oldBeta[i]);
            maxDiff = Math.max(maxDiff, diff);
        }
        
        return maxDiff < tolerance;
    }
}
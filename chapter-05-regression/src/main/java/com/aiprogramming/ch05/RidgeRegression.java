package com.aiprogramming.ch05;

import java.util.*;

/**
 * Ridge Regression Implementation with L2 Regularization
 */
public class RidgeRegression implements Regressor {
    
    private final double alpha;
    private Map<String, Double> coefficients;
    private double intercept;
    private List<String> featureNames;
    private boolean trained;
    
    public RidgeRegression(double alpha) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
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
        
        // X matrix (with bias column)
        double[][] X = new double[numSamples][numFeatures + 1];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            RegressionDataPoint point = trainingData.get(i);
            X[i][0] = 1.0; // Bias term
            
            for (int j = 0; j < numFeatures; j++) {
                String featureName = featureNames.get(j);
                X[i][j + 1] = point.getFeatures().getOrDefault(featureName, 0.0);
            }
            
            y[i] = point.getTarget();
        }
        
        // Ridge regression: β = (X'X + αI)^(-1)X'y
        double[][] XTranspose = transpose(X);
        double[][] XTX = multiply(XTranspose, X);
        
        // Add regularization term (don't regularize intercept)
        for (int i = 1; i < XTX.length; i++) {
            XTX[i][i] += alpha;
        }
        
        double[][] XTXInverse = inverse(XTX);
        double[] XTy = multiplyVector(XTranspose, y);
        double[] beta = multiplyVector(XTXInverse, XTy);
        
        // Extract intercept and coefficients
        this.intercept = beta[0];
        this.coefficients.clear();
        
        for (int i = 0; i < numFeatures; i++) {
            coefficients.put(featureNames.get(i), beta[i + 1]);
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
    
    @Override
    public String getName() {
        return "RidgeRegression(alpha=" + alpha + ")";
    }
    
    // Matrix operations (same as LinearRegression)
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    private double[][] multiply(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }
    
    private double[] multiplyVector(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        return result;
    }
    
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            double maxElement = Math.abs(augmented[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > maxElement) {
                    maxElement = Math.abs(augmented[k][i]);
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[maxRow];
                augmented[maxRow] = temp;
            }
            
            // Make diagonal element 1
            double pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new RuntimeException("Matrix is singular");
            }
            
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        return inverse;
    }
}
package com.aiprogramming.ch05;

import java.util.*;

/**
 * Support Vector Regression (SVR) Implementation with RBF Kernel
 */
public class SupportVectorRegression implements Regressor {
    
    private final double C;               // Regularization parameter
    private final double epsilon;         // Epsilon-tube parameter
    private final double gamma;           // RBF kernel parameter
    private final int maxIterations;
    private final double tolerance;
    
    private List<RegressionDataPoint> supportVectors;
    private double[] alphas;
    private double[] alphaStar;
    private double bias;
    private List<String> featureNames;
    private boolean trained;
    
    public SupportVectorRegression(double C, double epsilon, double gamma) {
        this(C, epsilon, gamma, 1000, 1e-6);
    }
    
    public SupportVectorRegression(double C, double epsilon, double gamma, 
                                 int maxIterations, double tolerance) {
        if (C <= 0 || epsilon < 0 || gamma <= 0) {
            throw new IllegalArgumentException("C, gamma must be positive, epsilon must be non-negative");
        }
        
        this.C = C;
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        this.featureNames = new ArrayList<>(trainingData.get(0).getFeatureNames());
        int numSamples = trainingData.size();
        
        // Initialize Lagrange multipliers
        this.alphas = new double[numSamples];
        this.alphaStar = new double[numSamples];
        this.bias = 0.0;
        
        // Simplified SMO-like algorithm for SVR
        // For a production implementation, you would use a more sophisticated optimizer
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            
            for (int i = 0; i < numSamples; i++) {
                double prediction = predictInternal(trainingData.get(i), trainingData);
                double target = trainingData.get(i).getTarget();
                double error = prediction - target;
                
                // Check KKT conditions and update if needed
                if (Math.abs(error) > epsilon + tolerance) {
                    // Find another example to update jointly
                    int j = findSecondExample(i, trainingData, error);
                    if (j != -1) {
                        changed |= updateAlphaPair(i, j, trainingData);
                    }
                }
            }
            
            if (!changed) {
                break;
            }
        }
        
        // Store support vectors (examples with non-zero alphas)
        this.supportVectors = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            if (alphas[i] > tolerance || alphaStar[i] > tolerance) {
                supportVectors.add(trainingData.get(i));
            }
        }
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        RegressionDataPoint queryPoint = new RegressionDataPoint(features, 0.0);
        return predictInternal(queryPoint, supportVectors);
    }
    
    private double predictInternal(RegressionDataPoint queryPoint, List<RegressionDataPoint> trainingData) {
        double prediction = bias;
        
        for (int i = 0; i < trainingData.size(); i++) {
            double alpha = (i < alphas.length) ? alphas[i] : 0.0;
            double alphaSt = (i < alphaStar.length) ? alphaStar[i] : 0.0;
            
            if (alpha > tolerance || alphaSt > tolerance) {
                double kernel = rbfKernel(queryPoint, trainingData.get(i));
                prediction += (alpha - alphaSt) * kernel;
            }
        }
        
        return prediction;
    }
    
    private double rbfKernel(RegressionDataPoint p1, RegressionDataPoint p2) {
        double squaredDistance = 0.0;
        
        for (String feature : featureNames) {
            double v1 = p1.getFeatures().getOrDefault(feature, 0.0);
            double v2 = p2.getFeatures().getOrDefault(feature, 0.0);
            double diff = v1 - v2;
            squaredDistance += diff * diff;
        }
        
        return Math.exp(-gamma * squaredDistance);
    }
    
    private int findSecondExample(int i, List<RegressionDataPoint> data, double error) {
        // Simple heuristic: find example with largest error of opposite sign
        int bestJ = -1;
        double bestError = 0.0;
        
        for (int j = 0; j < data.size(); j++) {
            if (j == i) continue;
            
            double prediction = predictInternal(data.get(j), data);
            double target = data.get(j).getTarget();
            double currentError = prediction - target;
            
            // Look for error of opposite sign with larger magnitude
            if (error * currentError < 0 && Math.abs(currentError) > Math.abs(bestError)) {
                bestJ = j;
                bestError = currentError;
            }
        }
        
        return bestJ;
    }
    
    private boolean updateAlphaPair(int i, int j, List<RegressionDataPoint> data) {
        // Simplified alpha update - in practice, this would be more complex
        double eta = 2 * rbfKernel(data.get(i), data.get(j)) - 
                    rbfKernel(data.get(i), data.get(i)) - 
                    rbfKernel(data.get(j), data.get(j));
        
        if (eta >= 0) return false;
        
        // Calculate bounds
        double L, H;
        double yi = data.get(i).getTarget();
        double yj = data.get(j).getTarget();
        
        // Simplified bounds calculation
        L = Math.max(0, alphas[j] - alphas[i]);
        H = Math.min(C, C + alphas[j] - alphas[i]);
        
        if (L >= H) return false;
        
        // Update alphas (simplified)
        double oldAlphaI = alphas[i];
        double oldAlphaJ = alphas[j];
        
        double predI = predictInternal(data.get(i), data);
        double predJ = predictInternal(data.get(j), data);
        
        double errorI = predI - yi;
        double errorJ = predJ - yj;
        
        alphas[j] = oldAlphaJ - (errorJ - errorI) / eta;
        alphas[j] = Math.max(L, Math.min(H, alphas[j]));
        
        if (Math.abs(alphas[j] - oldAlphaJ) < tolerance) {
            return false;
        }
        
        alphas[i] = oldAlphaI + (oldAlphaJ - alphas[j]);
        
        // Update bias (simplified)
        double deltaAlphaI = alphas[i] - oldAlphaI;
        double deltaAlphaJ = alphas[j] - oldAlphaJ;
        
        double b1 = bias - errorI - deltaAlphaI * rbfKernel(data.get(i), data.get(i)) - 
                   deltaAlphaJ * rbfKernel(data.get(i), data.get(j));
        
        double b2 = bias - errorJ - deltaAlphaI * rbfKernel(data.get(i), data.get(j)) - 
                   deltaAlphaJ * rbfKernel(data.get(j), data.get(j));
        
        if (0 < alphas[i] && alphas[i] < C) {
            bias = b1;
        } else if (0 < alphas[j] && alphas[j] < C) {
            bias = b2;
        } else {
            bias = (b1 + b2) / 2;
        }
        
        return true;
    }
    
    /**
     * Gets the regularization parameter C
     */
    public double getC() {
        return C;
    }
    
    /**
     * Gets the epsilon parameter
     */
    public double getEpsilon() {
        return epsilon;
    }
    
    /**
     * Gets the gamma parameter
     */
    public double getGamma() {
        return gamma;
    }
    
    /**
     * Gets the number of support vectors
     */
    public int getNumSupportVectors() {
        return trained ? supportVectors.size() : 0;
    }
    
    /**
     * Gets the support vector ratio
     */
    public double getSupportVectorRatio() {
        if (!trained || alphas == null) return 0.0;
        return (double) getNumSupportVectors() / alphas.length;
    }
    
    /**
     * Gets the support vectors
     */
    public List<RegressionDataPoint> getSupportVectors() {
        return trained ? new ArrayList<>(supportVectors) : new ArrayList<>();
    }
}
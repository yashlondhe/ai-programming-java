package com.aiprogramming.ch05;

import java.util.*;

/**
 * Polynomial Regression Implementation
 */
public class PolynomialRegression implements Regressor {
    
    private final int degree;
    private LinearRegression linearRegressor;
    private List<String> originalFeatures;
    private List<String> polynomialFeatures;
    private boolean trained;
    
    public PolynomialRegression(int degree) {
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be at least 1");
        }
        this.degree = degree;
        this.linearRegressor = new LinearRegression();
        this.originalFeatures = new ArrayList<>();
        this.polynomialFeatures = new ArrayList<>();
        this.trained = false;
    }
    
    @Override
    public void train(List<RegressionDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract original feature names
        this.originalFeatures = new ArrayList<>(trainingData.get(0).getFeatureNames());
        
        // Generate polynomial features
        this.polynomialFeatures = generatePolynomialFeatureNames(originalFeatures, degree);
        
        // Transform training data to include polynomial features
        List<RegressionDataPoint> transformedData = transformData(trainingData);
        
        // Train linear regressor on transformed data
        linearRegressor.train(transformedData);
        
        this.trained = true;
    }
    
    @Override
    public double predict(Map<String, Double> features) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        // Transform features to include polynomial terms
        Map<String, Double> transformedFeatures = transformFeatures(features);
        
        // Use linear regressor for prediction
        return linearRegressor.predict(transformedFeatures);
    }
    
    /**
     * Gets the degree of the polynomial
     */
    public int getDegree() {
        return degree;
    }
    
    /**
     * Gets the underlying linear regressor coefficients
     */
    public Map<String, Double> getCoefficients() {
        return linearRegressor.getCoefficients();
    }
    
    /**
     * Gets the intercept
     */
    public double getIntercept() {
        return linearRegressor.getIntercept();
    }
    
    private List<String> generatePolynomialFeatureNames(List<String> features, int degree) {
        List<String> polyFeatures = new ArrayList<>();
        
        // Generate all combinations of features up to the specified degree
        generatePolynomialCombinations(features, degree, "", 0, 0, polyFeatures);
        
        return polyFeatures;
    }
    
    private void generatePolynomialCombinations(List<String> features, int maxDegree, 
                                              String current, int currentDegree, 
                                              int startFeature, List<String> result) {
        if (currentDegree > 0) {
            result.add(current);
        }
        
        if (currentDegree < maxDegree) {
            for (int i = startFeature; i < features.size(); i++) {
                String newFeature = current.isEmpty() ? features.get(i) : current + "*" + features.get(i);
                generatePolynomialCombinations(features, maxDegree, newFeature, 
                                             currentDegree + 1, i, result);
            }
        }
    }
    
    private List<RegressionDataPoint> transformData(List<RegressionDataPoint> data) {
        List<RegressionDataPoint> transformedData = new ArrayList<>();
        
        for (RegressionDataPoint point : data) {
            Map<String, Double> transformedFeatures = transformFeatures(point.getFeatures());
            transformedData.add(new RegressionDataPoint(transformedFeatures, point.getTarget()));
        }
        
        return transformedData;
    }
    
    private Map<String, Double> transformFeatures(Map<String, Double> features) {
        Map<String, Double> transformedFeatures = new HashMap<>();
        
        for (String polyFeature : polynomialFeatures) {
            double value = calculatePolynomialFeatureValue(polyFeature, features);
            transformedFeatures.put(polyFeature, value);
        }
        
        return transformedFeatures;
    }
    
    private double calculatePolynomialFeatureValue(String polyFeature, Map<String, Double> features) {
        String[] featureParts = polyFeature.split("\\*");
        double value = 1.0;
        
        for (String part : featureParts) {
            value *= features.getOrDefault(part, 0.0);
        }
        
        return value;
    }
}
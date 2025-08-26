package com.aiprogramming.ch05;

import java.util.*;

/**
 * Simple test to debug regression algorithms
 */
public class SimpleRegressionTest {
    
    public static void main(String[] args) {
        System.out.println("=== Simple Regression Test ===");
        
        // Create simple test data
        List<RegressionDataPoint> testData = createSimpleTestData();
        
        System.out.println("Test data size: " + testData.size());
        System.out.println("First data point: " + testData.get(0));
        
        // Test Linear Regression
        LinearRegression lr = new LinearRegression();
        lr.train(testData);
        
        System.out.println("Linear Regression trained successfully");
        System.out.println("Intercept: " + lr.getIntercept());
        System.out.println("Coefficients: " + lr.getCoefficients());
        
        // Test prediction
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("x", 5.0);
        
        double prediction = lr.predict(testFeatures);
        System.out.println("Prediction for x=5.0: " + prediction);
        
        // Test evaluation
        RegressionEvaluator evaluator = new RegressionEvaluator();
        List<Double> actuals = new ArrayList<>();
        List<Double> predictions = new ArrayList<>();
        
        for (RegressionDataPoint point : testData) {
            actuals.add(point.getTarget());
            predictions.add(lr.predict(point.getFeatures()));
        }
        
        RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
        System.out.println("MAE: " + metrics.getMae());
        System.out.println("MSE: " + metrics.getMse());
        System.out.println("R²: " + metrics.getR2());
    }
    
    private static List<RegressionDataPoint> createSimpleTestData() {
        List<RegressionDataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 100; i++) {
            Map<String, Double> features = new HashMap<>();
            double x = random.nextDouble() * 10; // x between 0 and 10
            features.put("x", x);
            
            // Simple linear relationship: y = 2*x + 1 + noise
            double y = 2 * x + 1 + random.nextGaussian() * 0.5;
            
            data.add(new RegressionDataPoint(features, y));
        }
        
        return data;
    }
}

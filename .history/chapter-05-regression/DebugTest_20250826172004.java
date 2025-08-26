package com.aiprogramming.ch05;

import java.util.*;

public class DebugTest {
    public static void main(String[] args) {
        System.out.println("=== Debug Test ===");
        
        // Create simple test data
        List<RegressionDataPoint> testData = new ArrayList<>();
        
        // Add some simple test points
        Map<String, Double> features1 = new HashMap<>();
        features1.put("x1", 1.0);
        features1.put("x2", 2.0);
        testData.add(new RegressionDataPoint(features1, 5.0));
        
        Map<String, Double> features2 = new HashMap<>();
        features2.put("x1", 2.0);
        features2.put("x2", 3.0);
        testData.add(new RegressionDataPoint(features2, 8.0));
        
        Map<String, Double> features3 = new HashMap<>();
        features3.put("x1", 3.0);
        features3.put("x2", 4.0);
        testData.add(new RegressionDataPoint(features3, 11.0));
        
        System.out.println("Test data created with " + testData.size() + " points");
        
        // Test linear regression
        try {
            LinearRegression lr = new LinearRegression();
            lr.train(testData);
            
            System.out.println("Linear regression trained successfully");
            System.out.println("Intercept: " + lr.getIntercept());
            System.out.println("Coefficients: " + lr.getCoefficients());
            
            // Test prediction
            Map<String, Double> testFeatures = new HashMap<>();
            testFeatures.put("x1", 4.0);
            testFeatures.put("x2", 5.0);
            double prediction = lr.predict(testFeatures);
            System.out.println("Prediction for (4,5): " + prediction);
            
        } catch (Exception e) {
            System.out.println("Error in linear regression: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Test evaluator
        try {
            List<Double> actuals = Arrays.asList(5.0, 8.0, 11.0);
            List<Double> predictions = Arrays.asList(5.1, 8.2, 10.9);
            
            RegressionEvaluator evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
            
            System.out.println("Evaluation successful:");
            System.out.println("MAE: " + metrics.getMAE());
            System.out.println("MSE: " + metrics.getMSE());
            System.out.println("R²: " + metrics.getR2());
            
        } catch (Exception e) {
            System.out.println("Error in evaluation: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

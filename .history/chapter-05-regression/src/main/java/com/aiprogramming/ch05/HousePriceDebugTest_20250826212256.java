package com.aiprogramming.ch05;

import java.util.*;

/**
 * Debug test for house price prediction without normalization
 */
public class HousePriceDebugTest {
    
    public static void main(String[] args) {
        System.out.println("=== House Price Debug Test (No Normalization) ===");
        
        // Generate house data
        List<RegressionDataPoint> houseData = generateHouseData(100);
        
        System.out.println("House data size: " + houseData.size());
        System.out.println("First data point: " + houseData.get(0));
        
        // Split data
        RegressionDataSplitter splitter = new RegressionDataSplitter(0.8);
        RegressionDataSplit split = splitter.split(houseData);
        
        System.out.println("Training samples: " + split.getTrainingData().size());
        System.out.println("Test samples: " + split.getTestData().size());
        
        // Test Linear Regression without normalization
        LinearRegression lr = new LinearRegression();
        lr.train(split.getTrainingData());
        
        System.out.println("Linear Regression trained successfully");
        System.out.println("Intercept: " + lr.getIntercept());
        System.out.println("Coefficients: " + lr.getCoefficients());
        
        // Test prediction
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("sqft", 2000.0);
        testFeatures.put("bedrooms", 3.0);
        testFeatures.put("bathrooms", 2.0);
        testFeatures.put("age", 10.0);
        testFeatures.put("lot_size", 0.5);
        testFeatures.put("garage_size", 2.0);
        
        double prediction = lr.predict(testFeatures);
        System.out.println("Prediction for test house: " + prediction);
        
        // Test evaluation
        RegressionEvaluator evaluator = new RegressionEvaluator();
        List<Double> actuals = new ArrayList<>();
        List<Double> predictions = new ArrayList<>();
        
        for (RegressionDataPoint point : split.getTestData()) {
            actuals.add(point.getTarget());
            predictions.add(lr.predict(point.getFeatures()));
        }
        
        RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
        System.out.println("MAE: " + metrics.getMAE());
        System.out.println("MSE: " + metrics.getMSE());
        System.out.println("R²: " + metrics.getR2());
        
        // Check for NaN values
        System.out.println("Checking for NaN values:");
        System.out.println("Any NaN in actuals: " + actuals.stream().anyMatch(d -> Double.isNaN(d)));
        System.out.println("Any NaN in predictions: " + predictions.stream().anyMatch(d -> Double.isNaN(d)));
        System.out.println("Any NaN in coefficients: " + lr.getCoefficients().values().stream().anyMatch(d -> Double.isNaN(d)));
        System.out.println("Intercept is NaN: " + Double.isNaN(lr.getIntercept()));
    }
    
    /**
     * Generates synthetic house data for demonstration
     */
    private static List<RegressionDataPoint> generateHouseData(int numSamples) {
        List<RegressionDataPoint> data = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < numSamples; i++) {
            Map<String, Double> features = new HashMap<>();
            
            // Generate features
            double sqft = 1000 + random.nextGaussian() * 800; // Square footage
            double bedrooms = Math.max(1, 2 + random.nextGaussian() * 1.5); // Number of bedrooms
            double bathrooms = Math.max(1, 1.5 + random.nextGaussian() * 1.0); // Number of bathrooms
            double age = Math.max(0, random.nextGaussian() * 20); // Age of house in years
            double lotSize = 0.1 + random.nextGaussian() * 0.5; // Lot size in acres
            double garageSize = Math.max(0, random.nextGaussian() * 1.5); // Garage size
            
            features.put("sqft", sqft);
            features.put("bedrooms", bedrooms);
            features.put("bathrooms", bathrooms);
            features.put("age", age);
            features.put("lot_size", lotSize);
            features.put("garage_size", garageSize);
            
            // Calculate price based on realistic relationships
            double basePrice = 100000; // Base price
            double price = basePrice;
            
            // Square footage is the main driver
            price += sqft * 120; // $120 per sqft
            
            // Bedrooms and bathrooms add value
            price += bedrooms * 15000;
            price += bathrooms * 10000;
            
            // Age decreases value
            price -= age * 1000;
            
            // Lot size adds value
            price += lotSize * 50000;
            
            // Garage adds value
            price += garageSize * 8000;
            
            // Add some non-linear effects
            price += Math.pow(sqft / 1000, 1.5) * 20000; // Diminishing returns for very large houses
            price -= Math.pow(age / 50, 2) * 30000; // Accelerating depreciation for very old houses
            
            // Add noise
            price += random.nextGaussian() * 25000;
            
            // Ensure price is positive
            price = Math.max(price, 50000);
            
            data.add(new RegressionDataPoint(features, price));
        }
        
        return data;
    }
}

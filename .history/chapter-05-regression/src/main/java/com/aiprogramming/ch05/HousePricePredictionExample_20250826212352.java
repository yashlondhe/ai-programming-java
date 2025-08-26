package com.aiprogramming.ch05;

import java.util.*;

/**
 * Complete regression example: House Price Prediction
 */
public class HousePricePredictionExample {
    
    public static void main(String[] args) {
        System.out.println("=== House Price Prediction Example ===\n");
        
        // Generate sample house data
        List<RegressionDataPoint> houseData = generateHouseData(1000);
        
        // Split data into training and testing sets
        RegressionDataSplitter splitter = new RegressionDataSplitter(0.8);
        RegressionDataSplit split = splitter.split(houseData);
        
        System.out.println("Training samples: " + split.getTrainingData().size());
        System.out.println("Test samples: " + split.getTestData().size());
        
        // Preprocess data (normalize features)
        RegressionPreprocessor preprocessor = new RegressionPreprocessor();
        List<RegressionDataPoint> normalizedTrainData = preprocessor.normalizeFeatures(split.getTrainingData());
        List<RegressionDataPoint> normalizedTestData = preprocessor.normalizeFeatures(split.getTestData());
        
        // Train and evaluate different regressors
        List<Regressor> regressors = Arrays.asList(
            new LinearRegression(),
            new PolynomialRegression(2),
            new RidgeRegression(0.1),
            new RidgeRegression(1.0),
            new LassoRegression(0.1),
            new LassoRegression(1.0),
            new SupportVectorRegression(1.0, 0.1, 0.1)
        );
        
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (Regressor regressor : regressors) {
            System.out.println("\n" + regressor.getName() + " Results:");
            System.out.println("================================");
            
            // Train the regressor
            long trainStart = System.currentTimeMillis();
            regressor.train(normalizedTrainData);
            long trainTime = System.currentTimeMillis() - trainStart;
            
            // Make predictions
            long predictStart = System.currentTimeMillis();
            List<Double> predictions = new ArrayList<>();
            List<Double> actuals = new ArrayList<>();
            
            for (RegressionDataPoint testPoint : normalizedTestData) {
                predictions.add(regressor.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getTarget());
            }
            long predictTime = System.currentTimeMillis() - predictStart;
            
            // Evaluate performance
            RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
            metrics.printMetrics();
            
            System.out.printf("Training time: %d ms%n", trainTime);
            System.out.printf("Prediction time: %d ms%n", predictTime);
            
            // Print feature importance for applicable models
            if (regressor instanceof LinearRegression) {
                LinearRegression lr = (LinearRegression) regressor;
                System.out.println("Feature Importance:");
                lr.getFeatureImportance().entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .forEach(entry -> System.out.printf("  %s: %.3f%n", entry.getKey(), entry.getValue()));
            } else if (regressor instanceof LassoRegression) {
                LassoRegression lasso = (LassoRegression) regressor;
                System.out.printf("Selected Features: %d/%d%n", 
                    lasso.getSelectedFeatures().size(), lasso.getCoefficients().size());
                System.out.printf("Sparsity: %.1f%%%n", lasso.getSparsity() * 100);
                System.out.println("Selected Features: " + lasso.getSelectedFeatures());
            } else if (regressor instanceof SupportVectorRegression) {
                SupportVectorRegression svr = (SupportVectorRegression) regressor;
                System.out.printf("Support Vectors: %d (%.1f%%)%n", 
                    svr.getNumSupportVectors(), svr.getSupportVectorRatio() * 100);
            }
        }
        
        // Demonstrate overfitting with high-degree polynomial
        System.out.println("\n=== Overfitting Demonstration ===");
        demonstrateOverfitting(normalizedTrainData, normalizedTestData);
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
            
            // Add some non-linear effects (simplified to avoid NaN)
            price += Math.pow(Math.max(sqft / 1000, 0.1), 1.5) * 20000; // Diminishing returns for very large houses
            price -= Math.pow(Math.max(age / 50, 0.1), 2) * 30000; // Accelerating depreciation for very old houses
            
            // Add noise
            price += random.nextGaussian() * 25000;
            
            // Ensure price is positive and finite
            price = Math.max(price, 50000);
            if (Double.isNaN(price) || Double.isInfinite(price)) {
                price = 50000; // Fallback to base price
            }
            
            data.add(new RegressionDataPoint(features, price));
        }
        
        return data;
    }
    
    private static void demonstrateOverfitting(List<RegressionDataPoint> trainData, 
                                             List<RegressionDataPoint> testData) {
        
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (int degree = 1; degree <= 5; degree++) {
            PolynomialRegression poly = new PolynomialRegression(degree);
            poly.train(trainData);
            
            // Evaluate on training data
            List<Double> trainPredictions = new ArrayList<>();
            List<Double> trainActuals = new ArrayList<>();
            
            for (RegressionDataPoint point : trainData) {
                trainPredictions.add(poly.predict(point.getFeatures()));
                trainActuals.add(point.getTarget());
            }
            
            RegressionMetrics trainMetrics = evaluator.evaluate(trainActuals, trainPredictions);
            
            // Evaluate on test data
            List<Double> testPredictions = new ArrayList<>();
            List<Double> testActuals = new ArrayList<>();
            
            for (RegressionDataPoint point : testData) {
                testPredictions.add(poly.predict(point.getFeatures()));
                testActuals.add(point.getTarget());
            }
            
            RegressionMetrics testMetrics = evaluator.evaluate(testActuals, testPredictions);
            
            System.out.printf("Degree %d - Train R²: %.4f, Test R²: %.4f, Difference: %.4f%n",
                degree, trainMetrics.getR2(), testMetrics.getR2(), 
                trainMetrics.getR2() - testMetrics.getR2());
        }
    }
}
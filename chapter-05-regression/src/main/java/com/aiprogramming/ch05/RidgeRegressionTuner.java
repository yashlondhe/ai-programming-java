package com.aiprogramming.ch05;

import java.util.*;

/**
 * Hyperparameter tuning for Ridge Regression
 */
public class RidgeRegressionTuner {
    
    public static double findBestAlpha(List<RegressionDataPoint> data, 
                                      double[] alphaValues, int kFolds) {
        double bestAlpha = alphaValues[0];
        double bestScore = Double.NEGATIVE_INFINITY;
        
        System.out.println("=== Ridge Regression Hyperparameter Tuning ===");
        
        for (double alpha : alphaValues) {
            RidgeRegression ridge = new RidgeRegression(alpha);
            RegressionCrossValidationResults results = 
                RegressionCrossValidator.kFoldCrossValidation(ridge, data, kFolds);
            
            double meanR2 = results.getMeanR2();
            
            System.out.printf("Alpha: %.4f, Mean R²: %.4f ± %.4f%n", 
                alpha, meanR2, results.getStdR2());
            
            if (meanR2 > bestScore) {
                bestScore = meanR2;
                bestAlpha = alpha;
            }
        }
        
        System.out.printf("Best Alpha: %.4f with R²: %.4f%n", bestAlpha, bestScore);
        return bestAlpha;
    }
    
    public static void main(String[] args) {
        // Example usage
        System.out.println("=== Ridge Regression Tuning Example ===\n");
        
        // Generate sample data
        HousePricePredictionExample houseExample = new HousePricePredictionExample();
        List<RegressionDataPoint> data = generateSampleData(500);
        
        // Normalize data
        RegressionPreprocessor preprocessor = new RegressionPreprocessor();
        List<RegressionDataPoint> normalizedData = preprocessor.normalizeFeatures(data);
        
        // Define alpha values to try
        double[] alphaValues = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
        
        // Find best alpha using 5-fold cross-validation
        double bestAlpha = findBestAlpha(normalizedData, alphaValues, 5);
        
        // Train final model with best alpha
        RidgeRegression bestModel = new RidgeRegression(bestAlpha);
        bestModel.train(normalizedData);
        
        System.out.println("\nFinal model coefficients:");
        for (Map.Entry<String, Double> entry : bestModel.getCoefficients().entrySet()) {
            System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
        }
        System.out.printf("  Intercept: %.4f%n", bestModel.getIntercept());
    }
    
    private static List<RegressionDataPoint> generateSampleData(int numSamples) {
        List<RegressionDataPoint> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("feature1", random.nextGaussian());
            features.put("feature2", random.nextGaussian());
            features.put("feature3", random.nextGaussian());
            
            double target = 2.0 * features.get("feature1") + 
                           1.5 * features.get("feature2") + 
                           0.5 * features.get("feature3") + 
                           random.nextGaussian() * 0.1;
            
            data.add(new RegressionDataPoint(features, target));
        }
        
        return data;
    }
}
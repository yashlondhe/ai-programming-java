package com.aiprogramming.ch15;

import java.util.Arrays;
import java.util.List;

/**
 * Main example demonstrating time series analysis and forecasting techniques
 */
public class StockPricePredictionExample {
    
    public static void main(String[] args) {
        System.out.println("=== Time Series Analysis and Forecasting Examples ===\n");
        
        // Generate sample stock price data
        double[] stockPrices = generateSampleStockData(100);
        System.out.println("Generated sample stock price data with " + stockPrices.length + " points");
        
        // Example 1: Moving Averages
        demonstrateMovingAverages(stockPrices);
        
        // Example 2: Seasonal Decomposition
        demonstrateSeasonalDecomposition(stockPrices);
        
        // Example 3: ARIMA Forecasting
        demonstrateARIMAForecasting(stockPrices);
        
        // Example 4: LSTM Forecasting
        demonstrateLSTMForecasting(stockPrices);
        
        System.out.println("\n=== All examples completed successfully! ===");
    }
    
    /**
     * Generate sample stock price data with trend and noise
     */
    private static double[] generateSampleStockData(int length) {
        double[] data = new double[length];
        double basePrice = 100.0;
        double trend = 0.5;
        
        for (int i = 0; i < length; i++) {
            // Add trend
            double trendComponent = basePrice + trend * i;
            
            // Add seasonal component (weekly pattern)
            double seasonalComponent = 5.0 * Math.sin(2 * Math.PI * i / 7);
            
            // Add random noise
            double noise = (Math.random() - 0.5) * 10;
            
            data[i] = trendComponent + seasonalComponent + noise;
        }
        
        return data;
    }
    
    /**
     * Demonstrate various moving average techniques
     */
    private static void demonstrateMovingAverages(double[] data) {
        System.out.println("\n--- Moving Averages Demonstration ---");
        
        // Simple Moving Average
        double[] sma = MovingAverage.simpleMovingAverage(data, 10);
        System.out.println("Simple Moving Average (window=10): " + 
                          String.format("Last 5 values: [%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      sma[sma.length-5], sma[sma.length-4], sma[sma.length-3], 
                                      sma[sma.length-2], sma[sma.length-1]));
        
        // Exponential Moving Average
        double[] ema = MovingAverage.exponentialMovingAverage(data, 0.1);
        System.out.println("Exponential Moving Average (alpha=0.1): " + 
                          String.format("Last 5 values: [%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      ema[ema.length-5], ema[ema.length-4], ema[ema.length-3], 
                                      ema[ema.length-2], ema[ema.length-1]));
        
        // Weighted Moving Average
        double[] wma = MovingAverage.weightedMovingAverage(data, 10);
        System.out.println("Weighted Moving Average (window=10): " + 
                          String.format("Last 5 values: [%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      wma[wma.length-5], wma[wma.length-4], wma[wma.length-3], 
                                      wma[wma.length-2], wma[wma.length-1]));
        
        // Calculate optimal alpha
        double optimalAlpha = MovingAverage.calculateOptimalAlpha(data);
        System.out.println("Optimal alpha for EMA: " + String.format("%.3f", optimalAlpha));
    }
    
    /**
     * Demonstrate seasonal decomposition
     */
    private static void demonstrateSeasonalDecomposition(double[] data) {
        System.out.println("\n--- Seasonal Decomposition Demonstration ---");
        
        // Classical decomposition
        SeasonalDecomposition.DecompositionResult result = 
            SeasonalDecomposition.classicalDecomposition(data, 7); // Weekly seasonality
        
        double[] trend = result.getTrend();
        double[] seasonal = result.getSeasonal();
        double[] residual = result.getResidual();
        
        System.out.println("Classical Decomposition Results:");
        System.out.println("  Trend component - Last 5 values: " + 
                          String.format("[%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      trend[trend.length-5], trend[trend.length-4], 
                                      trend[trend.length-3], trend[trend.length-2], 
                                      trend[trend.length-1]));
        System.out.println("  Seasonal component - Last 5 values: " + 
                          String.format("[%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      seasonal[seasonal.length-5], seasonal[seasonal.length-4], 
                                      seasonal[seasonal.length-3], seasonal[seasonal.length-2], 
                                      seasonal[seasonal.length-1]));
        System.out.println("  Residual component - Last 5 values: " + 
                          String.format("[%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      residual[residual.length-5], residual[residual.length-4], 
                                      residual[residual.length-3], residual[residual.length-2], 
                                      residual[residual.length-1]));
        
        // Calculate variance explained by each component
        double totalVariance = calculateVariance(data);
        double trendVariance = calculateVariance(trend);
        double seasonalVariance = calculateVariance(seasonal);
        double residualVariance = calculateVariance(residual);
        
        System.out.println("Variance explained:");
        System.out.println("  Trend: " + String.format("%.1f%%", (trendVariance / totalVariance) * 100));
        System.out.println("  Seasonal: " + String.format("%.1f%%", (seasonalVariance / totalVariance) * 100));
        System.out.println("  Residual: " + String.format("%.1f%%", (residualVariance / totalVariance) * 100));
    }
    
    /**
     * Demonstrate ARIMA forecasting
     */
    private static void demonstrateARIMAForecasting(double[] data) {
        System.out.println("\n--- ARIMA Forecasting Demonstration ---");
        
        // Create and fit ARIMA model
        ARIMAModel arima = new ARIMAModel(1, 1, 1); // ARIMA(1,1,1)
        arima.fit(data);
        
        System.out.println("Fitted ARIMA model: " + arima);
        System.out.println("AR coefficients: " + Arrays.toString(arima.getARCoefficients()));
        System.out.println("MA coefficients: " + Arrays.toString(arima.getMACoefficients()));
        
        // Forecast next 10 values
        double[] forecast = arima.forecast(10);
        System.out.println("ARIMA Forecast (next 10 values): " + 
                          String.format("[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      forecast[0], forecast[1], forecast[2], forecast[3], forecast[4],
                                      forecast[5], forecast[6], forecast[7], forecast[8], forecast[9]));
        
        // Try different ARIMA models
        System.out.println("\nComparing different ARIMA models:");
        int[][] models = {{0,1,1}, {1,1,0}, {1,1,1}, {2,1,2}};
        
        for (int[] model : models) {
            ARIMAModel testModel = new ARIMAModel(model[0], model[1], model[2]);
            testModel.fit(data);
            System.out.println(String.format("ARIMA(%d,%d,%d) - AIC: %.2f, BIC: %.2f", 
                                           model[0], model[1], model[2], 
                                           testModel.calculateAIC(), testModel.calculateBIC()));
        }
    }
    
    /**
     * Demonstrate LSTM forecasting
     */
    private static void demonstrateLSTMForecasting(double[] data) {
        System.out.println("\n--- LSTM Forecasting Demonstration ---");
        
        // Normalize data
        double[] normalizedData = LSTMTimeSeries.normalize(data);
        
        // Create LSTM model
        LSTMTimeSeries lstm = new LSTMTimeSeries(1, 10, 1, 10);
        
        // Prepare training data
        List<LSTMTimeSeries.TrainingExample> trainingData = lstm.prepareTrainingData(normalizedData);
        System.out.println("Prepared " + trainingData.size() + " training examples");
        
        // Train the model (simplified training)
        System.out.println("Training LSTM model...");
        lstm.train(trainingData, 500, 0.01);
        
        // Prepare input sequence for prediction
        double[] inputSequence = new double[10];
        for (int i = 0; i < 10; i++) {
            inputSequence[i] = normalizedData[normalizedData.length - 10 + i];
        }
        
        // Make prediction
        double prediction = lstm.predict(inputSequence);
        
        // Denormalize prediction
        double min = Arrays.stream(data).min().orElse(0.0);
        double max = Arrays.stream(data).max().orElse(1.0);
        double denormalizedPrediction = LSTMTimeSeries.denormalize(new double[]{prediction}, min, max)[0];
        
        System.out.println("LSTM Prediction (next value): " + String.format("%.2f", denormalizedPrediction));
        System.out.println("Actual last value: " + String.format("%.2f", data[data.length - 1]));
        
        // Forecast multiple steps
        double[] forecast = lstm.forecast(inputSequence, 5);
        double[] denormalizedForecast = LSTMTimeSeries.denormalize(forecast, min, max);
        
        System.out.println("LSTM Forecast (next 5 values): " + 
                          String.format("[%.2f, %.2f, %.2f, %.2f, %.2f]", 
                                      denormalizedForecast[0], denormalizedForecast[1], 
                                      denormalizedForecast[2], denormalizedForecast[3], 
                                      denormalizedForecast[4]));
    }
    
    /**
     * Calculate variance of an array
     */
    private static double calculateVariance(double[] data) {
        double mean = Arrays.stream(data).average().orElse(0.0);
        return Arrays.stream(data)
                    .map(x -> Math.pow(x - mean, 2))
                    .average()
                    .orElse(0.0);
    }
}

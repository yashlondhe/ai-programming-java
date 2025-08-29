# Chapter 15: Time Series Analysis and Forecasting

This chapter covers time series analysis and forecasting techniques implemented in Java, including moving averages, seasonal decomposition, ARIMA models, and LSTM-based forecasting.

## Overview

Time series analysis is a statistical technique that deals with time-ordered data points. It's widely used in finance, economics, weather forecasting, and many other domains where data is collected over time.

## Key Concepts Covered

### 1. Time Series Fundamentals
- **Time Series**: A sequence of data points collected over time
- **Trend**: Long-term movement in the data
- **Seasonality**: Repeating patterns at regular intervals
- **Noise**: Random fluctuations in the data

### 2. Moving Averages
- **Simple Moving Average (SMA)**: Average of the last n values
- **Weighted Moving Average (WMA)**: Weighted average giving more importance to recent values
- **Exponential Moving Average (EMA)**: Exponentially weighted average
- **Double/Triple EMA**: Advanced smoothing techniques
- **Adaptive Moving Average (AMA)**: Adjusts smoothing based on volatility

### 3. Seasonal Decomposition
- **Classical Decomposition**: Separates time series into trend, seasonal, and residual components
- **Multiplicative Decomposition**: For data with increasing seasonal variation
- **X-13ARIMA-SEATS**: Advanced decomposition method

### 4. ARIMA Models
- **AutoRegressive (AR)**: Uses past values to predict future values
- **Integrated (I)**: Makes the series stationary through differencing
- **Moving Average (MA)**: Uses past forecast errors
- **Model Selection**: Using AIC and BIC criteria

### 5. LSTM for Time Series
- **Long Short-Term Memory**: Neural network architecture for sequence data
- **Sequence-to-Sequence**: Predicting future values from past sequences
- **Normalization**: Scaling data for better training

## Project Structure

```
src/main/java/com/aiprogramming/ch15/
├── TimeSeries.java                    # Core time series data structure
├── MovingAverage.java                 # Moving average implementations
├── SeasonalDecomposition.java         # Seasonal decomposition methods
├── ARIMAModel.java                    # ARIMA model implementation
├── LSTMTimeSeries.java                # LSTM-based forecasting
└── StockPricePredictionExample.java   # Main demonstration class
```

## Getting Started

### Prerequisites
- Java 11 or higher
- Maven 3.6 or higher
- Apache Commons Math library
- Apache Commons CSV library

### Building the Project

```bash
# Navigate to the chapter directory
cd chapter-15-time-series

# Compile the project
mvn clean compile

# Run the main example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch15.StockPricePredictionExample"
```

### Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=MovingAverageTest
```

## Usage Examples

### 1. Moving Averages

```java
// Simple Moving Average
double[] sma = MovingAverage.simpleMovingAverage(data, 10);

// Exponential Moving Average
double[] ema = MovingAverage.exponentialMovingAverage(data, 0.1);

// Weighted Moving Average
double[] wma = MovingAverage.weightedMovingAverage(data, 10);
```

### 2. Seasonal Decomposition

```java
// Classical decomposition
SeasonalDecomposition.DecompositionResult result = 
    SeasonalDecomposition.classicalDecomposition(data, 7);

double[] trend = result.getTrend();
double[] seasonal = result.getSeasonal();
double[] residual = result.getResidual();
```

### 3. ARIMA Forecasting

```java
// Create and fit ARIMA model
ARIMAModel arima = new ARIMAModel(1, 1, 1); // ARIMA(1,1,1)
arima.fit(data);

// Forecast next 10 values
double[] forecast = arima.forecast(10);

// Get model metrics
double aic = arima.calculateAIC();
double bic = arima.calculateBIC();
```

### 4. LSTM Forecasting

```java
// Create LSTM model
LSTMTimeSeries lstm = new LSTMTimeSeries(1, 10, 1, 10);

// Prepare training data
List<LSTMTimeSeries.TrainingExample> trainingData = 
    lstm.prepareTrainingData(normalizedData);

// Train the model
lstm.train(trainingData, 500, 0.01);

// Make predictions
double prediction = lstm.predict(inputSequence);
double[] forecast = lstm.forecast(inputSequence, 5);
```

## Key Features

### MovingAverage Class
- **Simple Moving Average**: Basic smoothing technique
- **Weighted Moving Average**: Emphasizes recent data points
- **Exponential Moving Average**: Smooths with exponential decay
- **Double/Triple EMA**: Reduces lag in trend detection
- **Adaptive Moving Average**: Adjusts to market volatility
- **Optimal Alpha Calculation**: Automatic parameter tuning

### SeasonalDecomposition Class
- **Classical Decomposition**: Additive model decomposition
- **Multiplicative Decomposition**: Multiplicative model decomposition
- **X-13ARIMA-SEATS**: Advanced decomposition method
- **Robust Trend Estimation**: Median-based trend calculation
- **Variance Analysis**: Component contribution analysis

### ARIMAModel Class
- **Yule-Walker Equations**: AR coefficient estimation
- **Innovation Algorithm**: MA coefficient estimation
- **Autocorrelation Analysis**: Statistical dependency analysis
- **Model Selection**: AIC and BIC criteria
- **Forecasting**: Multi-step ahead predictions

### LSTMTimeSeries Class
- **LSTM Architecture**: Long Short-Term Memory implementation
- **Sequence Processing**: Sliding window data preparation
- **Training**: Gradient-based optimization
- **Normalization**: Data scaling utilities
- **Multi-step Forecasting**: Recursive prediction

## Applications

### Financial Analysis
- Stock price prediction
- Market trend analysis
- Volatility forecasting
- Risk assessment

### Economic Forecasting
- GDP growth prediction
- Inflation forecasting
- Unemployment rate analysis
- Economic indicator modeling

### Weather Forecasting
- Temperature prediction
- Rainfall forecasting
- Climate pattern analysis
- Seasonal weather modeling

### Business Intelligence
- Sales forecasting
- Demand prediction
- Inventory optimization
- Customer behavior analysis

## Best Practices

### Data Preparation
1. **Handle Missing Values**: Use interpolation or forward-fill methods
2. **Outlier Detection**: Identify and handle extreme values
3. **Stationarity**: Ensure data is stationary for ARIMA models
4. **Normalization**: Scale data for neural network models

### Model Selection
1. **Data Characteristics**: Choose models based on data patterns
2. **Seasonality**: Use seasonal models for periodic data
3. **Trend Analysis**: Consider trend components in model selection
4. **Validation**: Use cross-validation for model evaluation

### Performance Evaluation
1. **Error Metrics**: Use MAE, MSE, RMSE for accuracy assessment
2. **Forecast Horizon**: Consider prediction timeframes
3. **Confidence Intervals**: Provide uncertainty estimates
4. **Backtesting**: Validate models on historical data

## Limitations and Considerations

### Moving Averages
- Lag in trend detection
- May miss sudden changes
- Window size selection is critical

### Seasonal Decomposition
- Assumes additive or multiplicative relationships
- Requires sufficient data for seasonal patterns
- Sensitive to outliers

### ARIMA Models
- Requires stationary data
- Parameter selection can be complex
- May not capture non-linear relationships

### LSTM Models
- Requires large amounts of training data
- Computationally intensive
- Black-box nature makes interpretation difficult

## Future Enhancements

1. **Prophet-like Models**: Facebook's Prophet implementation
2. **Deep Learning**: More advanced neural network architectures
3. **Ensemble Methods**: Combining multiple forecasting approaches
4. **Real-time Processing**: Streaming time series analysis
5. **Anomaly Detection**: Identifying unusual patterns in time series

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is part of the AI Programming with Java book and follows the same licensing terms.

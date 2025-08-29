# Chapter 15: Time Series Analysis and Forecasting

## Introduction

Time Series Analysis is a statistical technique that deals with time-ordered data points, where observations are collected at regular intervals over time. This field is fundamental in finance, economics, weather forecasting, and many other domains where understanding temporal patterns and predicting future values is crucial.

Time series data exhibits unique characteristics that distinguish it from other types of data:
- **Temporal Dependencies**: Each observation depends on previous observations
- **Trend**: Long-term movement in the data (increasing, decreasing, or stable)
- **Seasonality**: Repeating patterns at regular intervals
- **Cyclical Patterns**: Long-term oscillations without fixed periods
- **Noise**: Random fluctuations that cannot be explained by the model

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of time series analysis and its characteristics
- Implement various moving average techniques for smoothing and trend detection
- Apply seasonal decomposition methods to separate time series components
- Build and evaluate ARIMA models for time series forecasting
- Implement LSTM-based forecasting for complex temporal patterns
- Compare different forecasting approaches and select appropriate models
- Handle real-world time series data with missing values and outliers

### Key Concepts

- **Time Series**: A sequence of data points collected over time at regular intervals
- **Stationarity**: A time series whose statistical properties remain constant over time
- **Trend**: Long-term movement in the data (linear, exponential, or polynomial)
- **Seasonality**: Repeating patterns at fixed intervals (daily, weekly, monthly, yearly)
- **Autocorrelation**: Correlation between observations at different time lags
- **Forecasting**: Predicting future values based on historical patterns
- **Smoothing**: Reducing noise in time series data to reveal underlying patterns

## 15.1 Time Series Fundamentals

Time series analysis begins with understanding the basic structure and characteristics of temporal data. A time series can be decomposed into several components that help us understand and model the underlying patterns.

### 15.1.1 Time Series Components

A time series can be decomposed into four main components:

1. **Trend (T)**: Long-term movement in the data
2. **Seasonal (S)**: Repeating patterns at regular intervals
3. **Cyclical (C)**: Long-term oscillations without fixed periods
4. **Random/Noise (R)**: Unpredictable fluctuations

The relationship between these components can be:
- **Additive**: Y(t) = T(t) + S(t) + C(t) + R(t)
- **Multiplicative**: Y(t) = T(t) × S(t) × C(t) × R(t)

#### TimeSeries Class

```java
package com.aiprogramming.ch15;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Represents a time series with timestamps and corresponding values
 */
public class TimeSeries {
    private List<Long> timestamps;
    private List<Double> values;
    private String name;
    
    public TimeSeries(String name) {
        this.name = name;
        this.timestamps = new ArrayList<>();
        this.values = new ArrayList<>();
    }
    
    /**
     * Add a data point to the time series
     */
    public void addPoint(long timestamp, double value) {
        timestamps.add(timestamp);
        values.add(value);
    }
    
    /**
     * Get the value at a specific index
     */
    public double getValue(int index) {
        return values.get(index);
    }
    
    /**
     * Calculate the mean of all values
     */
    public double getMean() {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculate the standard deviation of all values
     */
    public double getStandardDeviation() {
        double mean = getMean();
        double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }
}
```

### 15.1.2 Stationarity

A time series is stationary if its statistical properties (mean, variance, autocorrelation) remain constant over time. Stationarity is crucial for many time series models, especially ARIMA.

**Tests for Stationarity:**
- **Augmented Dickey-Fuller Test**: Tests for unit roots
- **KPSS Test**: Tests for trend stationarity
- **Visual Inspection**: Plotting the series and its components

**Making a Series Stationary:**
- **Differencing**: Subtracting consecutive observations
- **Log Transformation**: For multiplicative relationships
- **Seasonal Differencing**: For seasonal patterns

## 15.2 Moving Averages

Moving averages are fundamental tools for smoothing time series data and identifying trends. They help reduce noise and reveal underlying patterns by averaging values over a specified window.

### 15.2.1 Simple Moving Average (SMA)

The Simple Moving Average calculates the average of the last n values:

SMA(t) = (Y(t) + Y(t-1) + ... + Y(t-n+1)) / n

#### Implementation

```java
package com.aiprogramming.ch15;

/**
 * Implements various moving average smoothing techniques for time series
 */
public class MovingAverage {
    
    /**
     * Simple Moving Average (SMA)
     * Calculates the average of the last n values
     */
    public static double[] simpleMovingAverage(double[] data, int windowSize) {
        if (windowSize <= 0 || windowSize > data.length) {
            throw new IllegalArgumentException("Invalid window size: " + windowSize);
        }
        
        double[] result = new double[data.length];
        
        // Fill the beginning with NaN
        for (int i = 0; i < windowSize - 1; i++) {
            result[i] = Double.NaN;
        }
        
        // Calculate SMA for the rest
        for (int i = windowSize - 1; i < data.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < windowSize; j++) {
                sum += data[i - j];
            }
            result[i] = sum / windowSize;
        }
        
        return result;
    }
}
```

**Advantages:**
- Simple to understand and implement
- Effective for smoothing random fluctuations
- Good for identifying trends

**Disadvantages:**
- Equal weight to all observations in the window
- Lag in trend detection
- May miss sudden changes

### 15.2.2 Weighted Moving Average (WMA)

The Weighted Moving Average gives more weight to recent observations:

WMA(t) = (n×Y(t) + (n-1)×Y(t-1) + ... + 1×Y(t-n+1)) / (n + (n-1) + ... + 1)

#### Implementation

```java
/**
 * Weighted Moving Average (WMA)
 * Gives more weight to recent values
 */
public static double[] weightedMovingAverage(double[] data, int windowSize) {
    if (windowSize <= 0 || windowSize > data.length) {
        throw new IllegalArgumentException("Invalid window size: " + windowSize);
    }
    
    double[] result = new double[data.length];
    
    // Fill the beginning with NaN
    for (int i = 0; i < windowSize - 1; i++) {
        result[i] = Double.NaN;
    }
    
    // Calculate WMA for the rest
    double weightSum = windowSize * (windowSize + 1) / 2.0; // Sum of weights
    
    for (int i = windowSize - 1; i < data.length; i++) {
        double weightedSum = 0.0;
        for (int j = 0; j < windowSize; j++) {
            weightedSum += (j + 1) * data[i - j];
        }
        result[i] = weightedSum / weightSum;
    }
    
    return result;
}
```

### 15.2.3 Exponential Moving Average (EMA)

The Exponential Moving Average gives exponentially decreasing weight to older observations:

EMA(t) = α × Y(t) + (1 - α) × EMA(t-1)

where α is the smoothing factor (0 < α < 1).

#### Implementation

```java
/**
 * Exponential Moving Average (EMA)
 * Gives exponentially decreasing weight to older values
 */
public static double[] exponentialMovingAverage(double[] data, double alpha) {
    if (alpha < 0 || alpha > 1) {
        throw new IllegalArgumentException("Alpha must be between 0 and 1: " + alpha);
    }
    
    double[] result = new double[data.length];
    
    // Initialize with the first value
    result[0] = data[0];
    
    // Calculate EMA for the rest
    for (int i = 1; i < data.length; i++) {
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
    }
    
    return result;
}
```

**Advantages:**
- Responds quickly to recent changes
- Less lag than SMA
- Smooths data effectively

**Disadvantages:**
- More sensitive to noise
- Requires parameter tuning (α)

### 15.2.4 Advanced Moving Averages

#### Double Exponential Moving Average (DEMA)

DEMA reduces lag by applying EMA twice:

DEMA(t) = 2 × EMA1(t) - EMA2(t)

where EMA2 is the EMA of EMA1.

#### Triple Exponential Moving Average (TEMA)

TEMA further reduces lag:

TEMA(t) = 3 × EMA1(t) - 3 × EMA2(t) + EMA3(t)

#### Adaptive Moving Average (AMA)

AMA adjusts smoothing based on market volatility:

```java
/**
 * Adaptive Moving Average (AMA)
 * Adjusts smoothing based on market volatility
 */
public static double[] adaptiveMovingAverage(double[] data, int fastPeriod, int slowPeriod) {
    if (fastPeriod >= slowPeriod) {
        throw new IllegalArgumentException("Fast period must be less than slow period");
    }
    
    double[] result = new double[data.length];
    result[0] = data[0];
    
    for (int i = 1; i < data.length; i++) {
        // Calculate efficiency ratio
        double change = Math.abs(data[i] - data[i - 1]);
        double volatility = 0.0;
        
        for (int j = Math.max(0, i - slowPeriod); j < i; j++) {
            volatility += Math.abs(data[j + 1] - data[j]);
        }
        
        double efficiencyRatio = volatility > 0 ? change / volatility : 0.0;
        
        // Calculate smoothing constant
        double fastSC = 2.0 / (fastPeriod + 1);
        double slowSC = 2.0 / (slowPeriod + 1);
        double amaSC = Math.pow(efficiencyRatio * (fastSC - slowSC) + slowSC, 2);
        
        // Apply AMA
        result[i] = amaSC * data[i] + (1 - amaSC) * result[i - 1];
    }
    
    return result;
}
```

## 15.3 Seasonal Decomposition

Seasonal decomposition separates a time series into its constituent components: trend, seasonal, and residual. This helps understand the underlying patterns and can improve forecasting accuracy.

### 15.3.1 Classical Decomposition

Classical decomposition assumes an additive model:

Y(t) = Trend(t) + Seasonal(t) + Residual(t)

#### Implementation

```java
package com.aiprogramming.ch15;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Implements seasonal decomposition of time series into trend, seasonal, and residual components
 */
public class SeasonalDecomposition {
    
    /**
     * Result of seasonal decomposition
     */
    public static class DecompositionResult {
        private double[] trend;
        private double[] seasonal;
        private double[] residual;
        private double[] original;
        
        public DecompositionResult(double[] original, double[] trend, double[] seasonal, double[] residual) {
            this.original = original;
            this.trend = trend;
            this.seasonal = seasonal;
            this.residual = residual;
        }
        
        public double[] getTrend() { return trend; }
        public double[] getSeasonal() { return seasonal; }
        public double[] getResidual() { return residual; }
        public double[] getOriginal() { return original; }
    }
    
    /**
     * Classical decomposition (additive model)
     * Y(t) = Trend(t) + Seasonal(t) + Residual(t)
     */
    public static DecompositionResult classicalDecomposition(double[] data, int period) {
        if (period <= 1 || period >= data.length / 2) {
            throw new IllegalArgumentException("Invalid period: " + period);
        }
        
        // Step 1: Calculate trend using moving average
        double[] trend = calculateTrend(data, period);
        
        // Step 2: Detrend the data
        double[] detrended = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            detrended[i] = data[i] - trend[i];
        }
        
        // Step 3: Calculate seasonal component
        double[] seasonal = calculateSeasonal(detrended, period);
        
        // Step 4: Calculate residuals
        double[] residual = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            residual[i] = detrended[i] - seasonal[i];
        }
        
        return new DecompositionResult(data, trend, seasonal, residual);
    }
}
```

**Steps in Classical Decomposition:**

1. **Trend Estimation**: Use centered moving average
2. **Detrending**: Subtract trend from original data
3. **Seasonal Estimation**: Calculate seasonal indices
4. **Residual Calculation**: Subtract seasonal from detrended data

### 15.3.2 Multiplicative Decomposition

For data with increasing seasonal variation, use multiplicative decomposition:

Y(t) = Trend(t) × Seasonal(t) × Residual(t)

#### Implementation

```java
/**
 * Multiplicative decomposition
 * Y(t) = Trend(t) * Seasonal(t) * Residual(t)
 */
public static DecompositionResult multiplicativeDecomposition(double[] data, int period) {
    if (period <= 1 || period >= data.length / 2) {
        throw new IllegalArgumentException("Invalid period: " + period);
    }
    
    // Step 1: Calculate trend using moving average
    double[] trend = calculateTrend(data, period);
    
    // Step 2: Detrend the data (division instead of subtraction)
    double[] detrended = new double[data.length];
    for (int i = 0; i < data.length; i++) {
        detrended[i] = trend[i] != 0 ? data[i] / trend[i] : 1.0;
    }
    
    // Step 3: Calculate seasonal component
    double[] seasonal = calculateSeasonal(detrended, period);
    
    // Step 4: Calculate residuals
    double[] residual = new double[data.length];
    for (int i = 0; i < data.length; i++) {
        residual[i] = seasonal[i] != 0 ? detrended[i] / seasonal[i] : 1.0;
    }
    
    return new DecompositionResult(data, trend, seasonal, residual);
}
```

### 15.3.3 X-13ARIMA-SEATS Decomposition

X-13ARIMA-SEATS is a more sophisticated decomposition method that uses advanced filtering techniques.

#### Implementation

```java
/**
 * X-13ARIMA-SEATS decomposition (simplified version)
 * Uses more sophisticated filtering and seasonal adjustment
 */
public static DecompositionResult x13Decomposition(double[] data, int period) {
    // This is a simplified version of X-13ARIMA-SEATS
    // In practice, you would use specialized libraries like RJDemetra
    
    // For now, we'll use a more robust trend calculation
    double[] trend = robustTrend(data, period);
    
    // Detrend and calculate seasonal
    double[] detrended = new double[data.length];
    for (int i = 0; i < data.length; i++) {
        detrended[i] = data[i] - trend[i];
    }
    
    double[] seasonal = robustSeasonal(detrended, period);
    
    // Calculate residuals
    double[] residual = new double[data.length];
    for (int i = 0; i < data.length; i++) {
        residual[i] = detrended[i] - seasonal[i];
    }
    
    return new DecompositionResult(data, trend, seasonal, residual);
}
```

## 15.4 ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models are powerful tools for time series forecasting. They combine autoregression, differencing, and moving average components.

### 15.4.1 ARIMA Components

ARIMA(p,d,q) has three parameters:
- **p**: Order of autoregression (AR)
- **d**: Degree of differencing (I)
- **q**: Order of moving average (MA)

#### ARIMA Model Implementation

```java
package com.aiprogramming.ch15;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Implements ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting
 */
public class ARIMAModel {
    private int p; // AR order
    private int d; // Differencing order
    private int q; // MA order
    private double[] arCoefficients;
    private double[] maCoefficients;
    private double[] residuals;
    private double[] originalData;
    private double[] differencedData;
    
    public ARIMAModel(int p, int d, int q) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.arCoefficients = new double[p];
        this.maCoefficients = new double[q];
    }
    
    /**
     * Fit the ARIMA model to the data
     */
    public void fit(double[] data) {
        this.originalData = data.clone();
        
        // Step 1: Apply differencing
        this.differencedData = applyDifferencing(data, d);
        
        // Step 2: Estimate AR coefficients
        if (p > 0) {
            this.arCoefficients = estimateARCoefficients(differencedData, p);
        }
        
        // Step 3: Estimate MA coefficients
        if (q > 0) {
            this.maCoefficients = estimateMACoefficients(differencedData, q);
        }
        
        // Step 4: Calculate residuals
        this.residuals = calculateResiduals();
    }
}
```

### 15.4.2 Autoregressive (AR) Component

The AR component uses past values to predict future values:

Y(t) = c + φ₁Y(t-1) + φ₂Y(t-2) + ... + φₚY(t-p) + ε(t)

#### AR Coefficient Estimation

```java
/**
 * Estimate AR coefficients using Yule-Walker equations
 */
private double[] estimateARCoefficients(double[] data, int order) {
    if (order == 0) return new double[0];
    
    // Calculate autocorrelations
    double[] autocorr = calculateAutocorrelations(data, order);
    
    // Build Toeplitz matrix
    RealMatrix toeplitz = new Array2DRowRealMatrix(order, order);
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            toeplitz.setEntry(i, j, autocorr[Math.abs(i - j)]);
        }
    }
    
    // Build right-hand side vector
    RealVector rhs = new ArrayRealVector(order);
    for (int i = 0; i < order; i++) {
        rhs.setEntry(i, autocorr[i + 1]);
    }
    
    // Solve Yule-Walker equations
    DecompositionSolver solver = new LUDecomposition(toeplitz).getSolver();
    RealVector solution = solver.solve(rhs);
    
    double[] coefficients = new double[order];
    for (int i = 0; i < order; i++) {
        coefficients[i] = solution.getEntry(i);
    }
    
    return coefficients;
}
```

### 15.4.3 Moving Average (MA) Component

The MA component uses past forecast errors:

Y(t) = μ + ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ... + θₚε(t-q)

#### MA Coefficient Estimation

```java
/**
 * Estimate MA coefficients using innovation algorithm
 */
private double[] estimateMACoefficients(double[] data, int order) {
    if (order == 0) return new double[0];
    
    // Initialize with zeros
    double[] coefficients = new double[order];
    
    // Simple estimation using autocorrelations
    double[] autocorr = calculateAutocorrelations(data, order);
    
    // Use first-order approximation
    if (order >= 1) {
        coefficients[0] = autocorr[1] / autocorr[0];
    }
    
    // For higher orders, use iterative approach
    for (int i = 1; i < order; i++) {
        coefficients[i] = autocorr[i + 1] / autocorr[0];
    }
    
    return coefficients;
}
```

### 15.4.4 Forecasting with ARIMA

#### Implementation

```java
/**
 * Forecast future values
 */
public double[] forecast(int steps) {
    if (differencedData == null) {
        throw new IllegalStateException("Model must be fitted before forecasting");
    }
    
    double[] forecast = new double[steps];
    double[] lastValues = new double[Math.max(p, q)];
    
    // Initialize with last known values
    for (int i = 0; i < lastValues.length; i++) {
        int idx = differencedData.length - lastValues.length + i;
        if (idx >= 0) {
            lastValues[i] = differencedData[idx];
        }
    }
    
    double[] lastResiduals = new double[q];
    for (int i = 0; i < q; i++) {
        int idx = residuals.length - q + i;
        if (idx >= 0) {
            lastResiduals[i] = residuals[idx];
        }
    }
    
    // Generate forecasts
    for (int t = 0; t < steps; t++) {
        double prediction = 0.0;
        
        // AR component
        for (int i = 0; i < p; i++) {
            prediction += arCoefficients[i] * lastValues[lastValues.length - 1 - i];
        }
        
        // MA component
        for (int i = 0; i < q; i++) {
            prediction += maCoefficients[i] * lastResiduals[lastResiduals.length - 1 - i];
        }
        
        forecast[t] = prediction;
        
        // Update last values
        for (int i = 0; i < lastValues.length - 1; i++) {
            lastValues[i] = lastValues[i + 1];
        }
        lastValues[lastValues.length - 1] = prediction;
        
        // Update residuals (assume zero for future)
        for (int i = 0; i < lastResiduals.length - 1; i++) {
            lastResiduals[i] = lastResiduals[i + 1];
        }
        lastResiduals[lastResiduals.length - 1] = 0.0;
    }
    
    // Apply inverse differencing
    return applyInverseDifferencing(forecast, d);
}
```

### 15.4.5 Model Selection

Model selection involves choosing the best p, d, q parameters. Common criteria include:

#### AIC (Akaike Information Criterion)

```java
/**
 * Calculate AIC (Akaike Information Criterion)
 */
public double calculateAIC() {
    if (residuals == null) {
        throw new IllegalStateException("Model must be fitted before calculating AIC");
    }
    
    int n = residuals.length;
    int k = p + q + 1; // Number of parameters + variance
    
    // Calculate residual sum of squares
    double rss = 0.0;
    for (double residual : residuals) {
        rss += residual * residual;
    }
    
    return n * Math.log(rss / n) + 2 * k;
}
```

#### BIC (Bayesian Information Criterion)

```java
/**
 * Calculate BIC (Bayesian Information Criterion)
 */
public double calculateBIC() {
    if (residuals == null) {
        throw new IllegalStateException("Model must be fitted before calculating BIC");
    }
    
    int n = residuals.length;
    int k = p + q + 1; // Number of parameters + variance
    
    // Calculate residual sum of squares
    double rss = 0.0;
    for (double residual : residuals) {
        rss += residual * residual;
    }
    
    return n * Math.log(rss / n) + k * Math.log(n);
}
```

## 15.5 LSTM for Time Series Forecasting

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network that can capture long-term dependencies in sequential data. They are particularly effective for time series forecasting.

### 15.5.1 LSTM Architecture

LSTM networks have three main gates:
- **Forget Gate**: Decides what information to discard
- **Input Gate**: Decides what new information to store
- **Output Gate**: Decides what information to output

#### LSTM Implementation

```java
package com.aiprogramming.ch15;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Simplified LSTM implementation for time series forecasting
 * This is a basic implementation - in practice, you would use libraries like DL4J or Weka
 */
public class LSTMTimeSeries {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private int sequenceLength;
    
    // LSTM parameters
    private double[][] Wf, Wi, Wo, Wc; // Weight matrices
    private double[] bf, bi, bo, bc;   // Bias vectors
    
    // Hidden state and cell state
    private double[] h;
    private double[] c;
    
    private Random random;
    
    public LSTMTimeSeries(int inputSize, int hiddenSize, int outputSize, int sequenceLength) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.sequenceLength = sequenceLength;
        
        this.random = new Random(42);
        
        initializeParameters();
    }
}
```

### 15.5.2 Forward Pass

#### Implementation

```java
/**
 * Forward pass through LSTM
 */
public double[] forward(double[] input) {
    // Concatenate input with previous hidden state
    double[] combined = concatenate(input, h);
    
    // Calculate gates
    double[] ft = sigmoid(elementWiseAdd(matrixVectorMultiply(Wf, combined), bf));
    double[] it = sigmoid(elementWiseAdd(matrixVectorMultiply(Wi, combined), bi));
    double[] ot = sigmoid(elementWiseAdd(matrixVectorMultiply(Wo, combined), bo));
    double[] c_tilde = tanh(elementWiseAdd(matrixVectorMultiply(Wc, combined), bc));
    
    // Update cell state
    c = elementWiseAdd(elementWiseMultiply(ft, c), elementWiseMultiply(it, c_tilde));
    
    // Update hidden state
    h = elementWiseMultiply(ot, tanh(c));
    
    return h.clone();
}
```

### 15.5.3 Training Data Preparation

#### Implementation

```java
/**
 * Prepare training data with sliding windows
 */
public List<TrainingExample> prepareTrainingData(double[] timeSeries) {
    List<TrainingExample> examples = new ArrayList<>();
    
    for (int i = 0; i <= timeSeries.length - sequenceLength - 1; i++) {
        double[] input = new double[sequenceLength];
        double target = timeSeries[i + sequenceLength];
        
        for (int j = 0; j < sequenceLength; j++) {
            input[j] = timeSeries[i + j];
        }
        
        examples.add(new TrainingExample(input, target));
    }
    
    return examples;
}
```

### 15.5.4 Training and Prediction

#### Implementation

```java
/**
 * Train the LSTM model
 */
public void train(List<TrainingExample> trainingData, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;
        
        for (TrainingExample example : trainingData) {
            // Reset hidden state and cell state
            h = new double[hiddenSize];
            c = new double[hiddenSize];
            
            // Forward pass through sequence
            for (int t = 0; t < example.input.length; t++) {
                double[] input = {example.input[t]};
                forward(input);
            }
            
            // Calculate loss (MSE)
            double prediction = h[0]; // Simple output mapping
            double loss = Math.pow(prediction - example.target, 2);
            totalLoss += loss;
            
            // Backward pass (simplified - in practice, you'd use backpropagation through time)
            // For this simplified version, we'll skip the complex backpropagation
        }
        
        if (epoch % 100 == 0) {
            System.out.printf("Epoch %d, Average Loss: %.6f%n", epoch, totalLoss / trainingData.size());
        }
    }
}

/**
 * Predict next value
 */
public double predict(double[] inputSequence) {
    // Reset hidden state and cell state
    h = new double[hiddenSize];
    c = new double[hiddenSize];
    
    // Forward pass through sequence
    for (int t = 0; t < inputSequence.length; t++) {
        double[] input = {inputSequence[t]};
        forward(input);
    }
    
    return h[0]; // Return first hidden state as prediction
}
```

### 15.5.5 Data Normalization

#### Implementation

```java
/**
 * Normalize data to [0, 1] range
 */
public static double[] normalize(double[] data) {
    double min = Double.MAX_VALUE;
    double max = Double.MIN_VALUE;
    
    for (double value : data) {
        min = Math.min(min, value);
        max = Math.max(max, value);
    }
    
    double[] normalized = new double[data.length];
    for (int i = 0; i < data.length; i++) {
        normalized[i] = (data[i] - min) / (max - min);
    }
    
    return normalized;
}

/**
 * Denormalize data back to original scale
 */
public static double[] denormalize(double[] normalizedData, double min, double max) {
    double[] denormalized = new double[normalizedData.length];
    for (int i = 0; i < normalizedData.length; i++) {
        denormalized[i] = normalizedData[i] * (max - min) + min;
    }
    return denormalized;
}
```

## 15.6 Practical Applications

### 15.6.1 Stock Price Prediction

Stock price prediction is one of the most common applications of time series analysis.

#### Implementation Example

```java
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
}
```

### 15.6.2 Weather Forecasting

Weather forecasting uses time series analysis to predict temperature, rainfall, and other meteorological variables.

### 15.6.3 Sales Forecasting

Businesses use time series analysis to forecast sales, demand, and inventory requirements.

### 15.6.4 Economic Forecasting

Economists use time series analysis to predict GDP, inflation, unemployment rates, and other economic indicators.

## 15.7 Model Evaluation and Selection

### 15.7.1 Error Metrics

Common error metrics for time series forecasting include:

#### Mean Absolute Error (MAE)
MAE = (1/n) × Σ|Y(t) - Ŷ(t)|

#### Mean Squared Error (MSE)
MSE = (1/n) × Σ(Y(t) - Ŷ(t))²

#### Root Mean Squared Error (RMSE)
RMSE = √MSE

#### Mean Absolute Percentage Error (MAPE)
MAPE = (1/n) × Σ|(Y(t) - Ŷ(t))/Y(t)| × 100

### 15.7.2 Cross-Validation for Time Series

Time series cross-validation requires special handling to avoid data leakage:

#### Time Series Split
- Use expanding window or rolling window approaches
- Never use future data to predict past values
- Maintain temporal order in validation

### 15.7.3 Model Comparison

When comparing different models:

1. **Use the same evaluation metric**
2. **Use the same test set**
3. **Consider computational complexity**
4. **Consider interpretability**
5. **Consider robustness to outliers**

## 15.8 Best Practices and Considerations

### 15.8.1 Data Preprocessing

1. **Handle Missing Values**
   - Forward fill, backward fill, or interpolation
   - Consider the nature of missingness

2. **Outlier Detection and Treatment**
   - Use statistical methods (IQR, Z-score)
   - Consider domain knowledge

3. **Stationarity**
   - Test for stationarity
   - Apply differencing if needed

4. **Seasonality Detection**
   - Use autocorrelation plots
   - Consider domain knowledge

### 15.8.2 Model Selection Guidelines

1. **For Short-term Forecasting**: Use simple models (SMA, EMA)
2. **For Trend Analysis**: Use moving averages or linear regression
3. **For Seasonal Data**: Use seasonal decomposition or SARIMA
4. **For Complex Patterns**: Use LSTM or other neural networks
5. **For Multiple Variables**: Use VAR or multivariate LSTM

### 15.8.3 Common Pitfalls

1. **Overfitting**: Using too many parameters
2. **Data Leakage**: Using future information to predict past
3. **Ignoring Seasonality**: Not accounting for periodic patterns
4. **Not Validating Assumptions**: Assuming stationarity without testing
5. **Ignoring External Factors**: Not considering exogenous variables

## 15.9 Advanced Topics

### 15.9.1 Prophet-like Models

Facebook's Prophet model combines trend, seasonality, and holiday effects:

```java
// Simplified Prophet-like model
public class ProphetModel {
    private double[] trend;
    private double[] seasonal;
    private double[] holiday;
    
    public double[] forecast(double[] data, int periods) {
        // Combine trend, seasonal, and holiday components
        double[] forecast = new double[periods];
        for (int i = 0; i < periods; i++) {
            forecast[i] = trend[i] + seasonal[i] + holiday[i];
        }
        return forecast;
    }
}
```

### 15.9.2 Ensemble Methods

Combining multiple forecasting models can improve accuracy:

```java
public class EnsembleForecaster {
    private List<ForecastingModel> models;
    
    public double[] ensembleForecast(double[] data, int periods) {
        double[] ensemble = new double[periods];
        
        for (ForecastingModel model : models) {
            double[] forecast = model.forecast(data, periods);
            for (int i = 0; i < periods; i++) {
                ensemble[i] += forecast[i];
            }
        }
        
        // Average the forecasts
        for (int i = 0; i < periods; i++) {
            ensemble[i] /= models.size();
        }
        
        return ensemble;
    }
}
```

### 15.9.3 Real-time Forecasting

For real-time applications, consider:
- Incremental learning
- Online algorithms
- Streaming data processing
- Model updating strategies

## 15.10 Summary

Time series analysis and forecasting are essential tools for understanding temporal patterns and making predictions. This chapter covered:

### Key Takeaways

1. **Moving Averages**: Fundamental tools for smoothing and trend detection
2. **Seasonal Decomposition**: Separating time series into components
3. **ARIMA Models**: Statistical approach to time series forecasting
4. **LSTM Networks**: Deep learning approach for complex temporal patterns
5. **Model Selection**: Choosing appropriate models based on data characteristics
6. **Evaluation**: Proper validation and error measurement

### Practical Skills

- Implement various moving average techniques
- Perform seasonal decomposition
- Build and evaluate ARIMA models
- Train LSTM networks for time series
- Compare different forecasting approaches
- Handle real-world time series data

### Next Steps

1. **Explore Advanced Models**: VAR, VECM, state space models
2. **Learn Deep Learning**: More sophisticated neural network architectures
3. **Study Multivariate Time Series**: Multiple variables and their interactions
4. **Practice with Real Data**: Apply techniques to actual datasets
5. **Consider Domain-specific Methods**: Finance, economics, meteorology

Time series analysis is a vast field with applications across many domains. The techniques covered in this chapter provide a solid foundation for understanding and forecasting temporal data. As you gain experience, you'll learn to choose the right tools for specific problems and develop intuition for time series patterns.

## Exercises

### Exercise 15.1: Moving Average Analysis
Implement a function that compares different moving average techniques on a given time series and visualizes the results.

### Exercise 15.2: Seasonal Decomposition
Create a program that performs seasonal decomposition on monthly sales data and analyzes the components.

### Exercise 15.3: ARIMA Model Selection
Build a function that automatically selects the best ARIMA model parameters using AIC and BIC criteria.

### Exercise 15.4: LSTM Forecasting
Implement a complete LSTM-based forecasting system with proper data preprocessing and evaluation.

### Exercise 15.5: Ensemble Forecasting
Create an ensemble method that combines multiple forecasting approaches and evaluates the improvement in accuracy.

### Exercise 15.6: Real-time Forecasting
Develop a system that can update forecasts in real-time as new data arrives.

### Exercise 15.7: Anomaly Detection
Implement anomaly detection in time series data using statistical methods and machine learning.

### Exercise 15.8: Multivariate Time Series
Extend the ARIMA model to handle multiple variables (VAR model) and compare with univariate approaches.

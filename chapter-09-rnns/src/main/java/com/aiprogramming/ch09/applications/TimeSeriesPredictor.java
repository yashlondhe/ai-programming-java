package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.RNNCell;

import java.util.*;

/**
 * Time series prediction using Recurrent Neural Networks.
 * 
 * This class implements a time series predictor that can forecast
 * future values based on historical data patterns.
 */
public class TimeSeriesPredictor {
    private RNNCell rnn;
    private int sequenceLength;
    private int hiddenSize;
    private double learningRate;
    private boolean isTrained;
    private double minValue;
    private double maxValue;
    
    /**
     * Constructor for TimeSeriesPredictor.
     */
    public TimeSeriesPredictor() {
        this.sequenceLength = 10;
        this.hiddenSize = 32;
        this.learningRate = 0.01;
        this.isTrained = false;
    }
    
    /**
     * Train the time series predictor on historical data.
     * 
     * @param data Historical time series data
     * @param epochs Number of training epochs
     */
    public void train(double[] data, int epochs) {
        if (data.length < sequenceLength + 1) {
            throw new IllegalArgumentException("Data must be longer than sequence length + 1");
        }
        
        // Normalize data
        normalizeData(data);
        
        // Initialize RNN
        rnn = new RNNCell(1, hiddenSize, 1);
        
        // Prepare training sequences
        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();
        
        for (int i = 0; i <= data.length - sequenceLength - 1; i++) {
            // Create input sequence
            double[] inputSequence = new double[sequenceLength];
            for (int j = 0; j < sequenceLength; j++) {
                inputSequence[j] = data[i + j];
            }
            
            // Target is the next value after the sequence
            double target = data[i + sequenceLength];
            
            inputs.add(inputSequence);
            targets.add(new double[]{target});
        }
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < inputs.size(); i++) {
                // Reset RNN state
                rnn.resetState();
                
                // Forward pass through sequence
                double[] inputSequence = inputs.get(i);
                List<double[]> outputs = new ArrayList<>();
                
                for (double input : inputSequence) {
                    double[] output = rnn.forward(new double[]{input});
                    outputs.add(output);
                }
                
                // Get final output and compute loss
                double[] finalOutput = outputs.get(outputs.size() - 1);
                double target = targets.get(i)[0];
                double loss = meanSquaredError(finalOutput[0], target);
                totalLoss += loss;
                
                // Simplified backpropagation (for demonstration)
                // In a real implementation, you would compute gradients and update weights
            }
            
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d, Average Loss: %.6f\n", epoch, totalLoss / inputs.size());
            }
        }
        
        isTrained = true;
    }
    
    /**
     * Predict next values in the time series.
     * 
     * @param historicalData Historical data points
     * @param predictionLength Number of future values to predict
     * @return Array of predicted values
     */
    public double[] predictNextValues(double[] historicalData, int predictionLength) {
        if (!isTrained || rnn == null) {
            throw new IllegalStateException("Model must be trained before making predictions");
        }
        
        if (historicalData.length < sequenceLength) {
            throw new IllegalArgumentException("Historical data must be at least " + sequenceLength + " points");
        }
        
        // Normalize historical data
        double[] normalizedData = normalizeData(historicalData);
        
        // Use the last sequenceLength points as initial sequence
        double[] initialSequence = new double[sequenceLength];
        System.arraycopy(normalizedData, normalizedData.length - sequenceLength, 
                        initialSequence, 0, sequenceLength);
        
        // Reset RNN state
        rnn.resetState();
        
        // Feed initial sequence to RNN
        for (double value : initialSequence) {
            rnn.forward(new double[]{value});
        }
        
        // Generate predictions
        double[] predictions = new double[predictionLength];
        for (int i = 0; i < predictionLength; i++) {
            double[] output = rnn.forward(new double[]{rnn.getHiddenState()[0]});
            predictions[i] = denormalizeValue(output[0]);
            
            // Use prediction as next input
            rnn.forward(new double[]{output[0]});
        }
        
        return predictions;
    }
    
    /**
     * Normalize data to [0, 1] range.
     * 
     * @param data Input data
     * @return Normalized data
     */
    private double[] normalizeData(double[] data) {
        // Find min and max values
        minValue = Double.MAX_VALUE;
        maxValue = Double.MIN_VALUE;
        
        for (double value : data) {
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }
        
        // Normalize to [0, 1]
        double[] normalized = new double[data.length];
        double range = maxValue - minValue;
        
        if (range == 0) {
            // All values are the same
            Arrays.fill(normalized, 0.5);
        } else {
            for (int i = 0; i < data.length; i++) {
                normalized[i] = (data[i] - minValue) / range;
            }
        }
        
        return normalized;
    }
    
    /**
     * Denormalize a value from [0, 1] range back to original scale.
     * 
     * @param normalizedValue Normalized value
     * @return Denormalized value
     */
    private double denormalizeValue(double normalizedValue) {
        return normalizedValue * (maxValue - minValue) + minValue;
    }
    
    /**
     * Compute mean squared error.
     * 
     * @param predicted Predicted value
     * @param target Target value
     * @return MSE
     */
    private double meanSquaredError(double predicted, double target) {
        double diff = predicted - target;
        return diff * diff;
    }
    
    /**
     * Predict a single next value.
     * 
     * @param historicalData Historical data points
     * @return Predicted next value
     */
    public double predictNextValue(double[] historicalData) {
        double[] predictions = predictNextValues(historicalData, 1);
        return predictions[0];
    }
    
    /**
     * Evaluate prediction accuracy using mean absolute error.
     * 
     * @param actual Actual values
     * @param predicted Predicted values
     * @return Mean absolute error
     */
    public double evaluateAccuracy(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Actual and predicted arrays must have same length");
        }
        
        double totalError = 0.0;
        for (int i = 0; i < actual.length; i++) {
            totalError += Math.abs(actual[i] - predicted[i]);
        }
        
        return totalError / actual.length;
    }
    
    /**
     * Evaluate prediction accuracy using root mean squared error.
     * 
     * @param actual Actual values
     * @param predicted Predicted values
     * @return Root mean squared error
     */
    public double evaluateRMSE(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Actual and predicted arrays must have same length");
        }
        
        double totalSquaredError = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double error = actual[i] - predicted[i];
            totalSquaredError += error * error;
        }
        
        return Math.sqrt(totalSquaredError / actual.length);
    }
    
    /**
     * Set sequence length for training.
     * 
     * @param sequenceLength New sequence length
     */
    public void setSequenceLength(int sequenceLength) {
        this.sequenceLength = sequenceLength;
    }
    
    /**
     * Set hidden size for the RNN.
     * 
     * @param hiddenSize New hidden size
     */
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    /**
     * Set learning rate for training.
     * 
     * @param learningRate New learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Get sequence length.
     * 
     * @return Current sequence length
     */
    public int getSequenceLength() {
        return sequenceLength;
    }
    
    /**
     * Get hidden size.
     * 
     * @return Current hidden size
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * Check if model is trained.
     * 
     * @return True if model has been trained
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Get data normalization range.
     * 
     * @return Array with [minValue, maxValue]
     */
    public double[] getNormalizationRange() {
        return new double[]{minValue, maxValue};
    }
}

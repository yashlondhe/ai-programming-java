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
    
    /**
     * Initialize LSTM parameters
     */
    private void initializeParameters() {
        // Initialize weight matrices with Xavier initialization
        Wf = new double[hiddenSize][inputSize + hiddenSize];
        Wi = new double[hiddenSize][inputSize + hiddenSize];
        Wo = new double[hiddenSize][inputSize + hiddenSize];
        Wc = new double[hiddenSize][inputSize + hiddenSize];
        
        double scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                Wf[i][j] = (random.nextDouble() - 0.5) * 2 * scale;
                Wi[i][j] = (random.nextDouble() - 0.5) * 2 * scale;
                Wo[i][j] = (random.nextDouble() - 0.5) * 2 * scale;
                Wc[i][j] = (random.nextDouble() - 0.5) * 2 * scale;
            }
        }
        
        // Initialize bias vectors
        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bo = new double[hiddenSize];
        bc = new double[hiddenSize];
        
        // Initialize hidden state and cell state
        h = new double[hiddenSize];
        c = new double[hiddenSize];
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Tanh activation function
     */
    private double tanh(double x) {
        return Math.tanh(x);
    }
    
    /**
     * Apply sigmoid to array
     */
    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = sigmoid(x[i]);
        }
        return result;
    }
    
    /**
     * Apply tanh to array
     */
    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = tanh(x[i]);
        }
        return result;
    }
    
    /**
     * Element-wise multiplication
     */
    private double[] elementWiseMultiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    /**
     * Element-wise addition
     */
    private double[] elementWiseAdd(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    /**
     * Matrix-vector multiplication
     */
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
    
    /**
     * Concatenate two arrays
     */
    private double[] concatenate(double[] a, double[] b) {
        double[] result = new double[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
    
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
    
    /**
     * Forecast multiple steps ahead
     */
    public double[] forecast(double[] inputSequence, int steps) {
        double[] forecast = new double[steps];
        double[] currentSequence = inputSequence.clone();
        
        for (int i = 0; i < steps; i++) {
            double prediction = predict(currentSequence);
            forecast[i] = prediction;
            
            // Update sequence for next prediction
            double[] newSequence = new double[currentSequence.length];
            System.arraycopy(currentSequence, 1, newSequence, 0, currentSequence.length - 1);
            newSequence[newSequence.length - 1] = prediction;
            currentSequence = newSequence;
        }
        
        return forecast;
    }
    
    /**
     * Training example class
     */
    public static class TrainingExample {
        public double[] input;
        public double target;
        
        public TrainingExample(double[] input, double target) {
            this.input = input;
            this.target = target;
        }
    }
    
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
}

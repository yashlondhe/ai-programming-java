# Chapter 9: Recurrent Neural Networks (RNNs)

## Introduction

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining internal memory through feedback connections. Unlike feedforward neural networks, RNNs can handle variable-length sequences and capture temporal dependencies, making them ideal for tasks like natural language processing, speech recognition, and time series forecasting.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of RNNs and their applications
- Implement basic RNN cells from scratch
- Build and train LSTM networks for sequence modeling
- Implement GRU networks as a more efficient alternative to LSTM
- Apply RNNs to text generation, sentiment analysis, and time series forecasting
- Handle the vanishing gradient problem in RNNs
- Use attention mechanisms for improved sequence modeling

### Key Concepts

- **Recurrent Neural Networks**: Neural networks with feedback connections for processing sequential data
- **Long Short-Term Memory (LSTM)**: Advanced RNN architecture that can learn long-term dependencies
- **Gated Recurrent Unit (GRU)**: Simplified LSTM variant with fewer parameters
- **Backpropagation Through Time (BPTT)**: Algorithm for training RNNs
- **Vanishing Gradient Problem**: Challenge in training deep RNNs
- **Attention Mechanisms**: Techniques for focusing on relevant parts of input sequences

## 9.1 Basic Recurrent Neural Networks

The basic RNN cell maintains a hidden state that is updated at each time step based on the current input and the previous hidden state.

### 9.1.1 RNN Architecture

A simple RNN computes:
- Hidden state: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
- Output: `y_t = W_hy * h_t + b_y`

#### Implementation

```java
package com.aiprogramming.ch09.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Basic Recurrent Neural Network (RNN) cell implementation.
 * 
 * The RNN cell maintains a hidden state that is updated at each time step
 * based on the current input and the previous hidden state.
 * 
 * Mathematical formulation:
 * h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
 * y_t = W_hy * h_t + b_y
 */
public class RNNCell {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    
    // Weight matrices
    private double[][] W_xh;  // Input to hidden weights
    private double[][] W_hh;  // Hidden to hidden weights
    private double[][] W_hy;  // Hidden to output weights
    
    // Bias vectors
    private double[] b_h;     // Hidden bias
    private double[] b_y;     // Output bias
    
    // Current hidden state
    private double[] hiddenState;
    
    /**
     * Constructor for RNN cell.
     * 
     * @param inputSize Size of input vectors
     * @param hiddenSize Size of hidden state
     * @param outputSize Size of output vectors
     */
    public RNNCell(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        initializeWeights();
        resetState();
    }
    
    /**
     * Forward pass for a single time step.
     * 
     * @param input Input vector for current time step
     * @return Output vector for current time step
     */
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize);
        }
        
        // Compute new hidden state: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
        double[] newHiddenState = new double[hiddenSize];
        
        // W_xh * x_t
        for (int i = 0; i < hiddenSize; i++) {
            newHiddenState[i] = b_h[i];  // Start with bias
            for (int j = 0; j < inputSize; j++) {
                newHiddenState[i] += W_xh[j][i] * input[j];
            }
        }
        
        // W_hh * h_{t-1} (if not first time step)
        if (hiddenState != null) {
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    newHiddenState[i] += W_hh[j][i] * hiddenState[j];
                }
            }
        }
        
        // Apply tanh activation
        for (int i = 0; i < hiddenSize; i++) {
            newHiddenState[i] = Math.tanh(newHiddenState[i]);
        }
        
        // Update hidden state
        hiddenState = newHiddenState;
        
        // Compute output: y_t = W_hy * h_t + b_y
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = b_y[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += W_hy[j][i] * hiddenState[j];
            }
        }
        
        return output;
    }
}
```

### 9.1.2 Backpropagation Through Time (BPTT)

BPTT is the algorithm used to train RNNs by unrolling the network through time and computing gradients.

#### Implementation

```java
/**
 * Backward pass through time (BPTT) for computing gradients.
 * 
 * @param outputGradients Gradients with respect to outputs
 * @return Gradients with respect to inputs
 */
public List<double[]> backward(List<double[]> outputGradients) {
    if (outputGradients.size() != outputs.size()) {
        throw new IllegalArgumentException("Output gradients size must match outputs size");
    }
    
    // Initialize gradient accumulators
    double[][] dW_xh = new double[inputSize][hiddenSize];
    double[][] dW_hh = new double[hiddenSize][hiddenSize];
    double[][] dW_hy = new double[hiddenSize][outputSize];
    double[] db_h = new double[hiddenSize];
    double[] db_y = new double[outputSize];
    
    // Initialize hidden state gradient
    double[] dh_next = new double[hiddenSize];
    
    // Backward pass through time
    for (int t = outputs.size() - 1; t >= 0; t--) {
        // Gradient with respect to output
        double[] dy = outputGradients.get(t);
        
        // Gradient with respect to W_hy and b_y
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dW_hy[i][j] += hiddenStates.get(t)[i] * dy[j];
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            db_y[j] += dy[j];
        }
        
        // Gradient with respect to hidden state
        double[] dh = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dh[i] += W_hy[i][j] * dy[j];
            }
        }
        
        // Add gradient from next time step
        for (int i = 0; i < hiddenSize; i++) {
            dh[i] += dh_next[i];
        }
        
        // Gradient with respect to tanh input
        double[] dh_raw = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double tanh_input = 0.0;
            
            // Recompute tanh input
            tanh_input += b_h[i];
            for (int j = 0; j < inputSize; j++) {
                tanh_input += W_xh[j][i] * inputs.get(t)[j];
            }
            
            if (t > 0) {
                for (int j = 0; j < hiddenSize; j++) {
                    tanh_input += W_hh[j][i] * hiddenStates.get(t-1)[j];
                }
            }
            
            // tanh derivative: 1 - tanh^2
            dh_raw[i] = dh[i] * (1.0 - Math.pow(Math.tanh(tanh_input), 2));
        }
        
        // Gradients with respect to weights and biases
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                dW_xh[i][j] += inputs.get(t)[i] * dh_raw[j];
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            db_h[i] += dh_raw[i];
        }
        
        if (t > 0) {
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dW_hh[i][j] += hiddenStates.get(t-1)[i] * dh_raw[j];
                }
            }
            
            // Gradient for next time step
            dh_next = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dh_next[i] += W_hh[i][j] * dh_raw[j];
                }
            }
        }
    }
    
    // Store gradients for weight updates
    this.dW_xh = dW_xh;
    this.dW_hh = dW_hh;
    this.dW_hy = dW_hy;
    this.db_h = db_h;
    this.db_y = db_y;
    
    return inputGradients;
}
```

## 9.2 Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem in RNNs by introducing a cell state and three gates that control information flow.

### 9.2.1 LSTM Architecture

LSTM uses three gates to control information flow:
- **Forget Gate**: `f_t = σ(W_f * [h_{t-1}, x_t] + b_f)`
- **Input Gate**: `i_t = σ(W_i * [h_{t-1}, x_t] + b_i)`
- **Output Gate**: `o_t = σ(W_o * [h_{t-1}, x_t] + b_o)`
- **Cell State**: `C_t = f_t * C_{t-1} + i_t * tanh(W_C * [h_{t-1}, x_t] + b_C)`
- **Hidden State**: `h_t = o_t * tanh(C_t)`

#### Implementation

```java
package com.aiprogramming.ch09.core;

/**
 * Long Short-Term Memory (LSTM) cell implementation.
 * 
 * LSTM addresses the vanishing gradient problem in RNNs by introducing
 * a cell state and three gates that control information flow.
 * 
 * Mathematical formulation:
 * f_t = σ(W_f * [h_{t-1}, x_t] + b_f)     // Forget gate
 * i_t = σ(W_i * [h_{t-1}, x_t] + b_i)     // Input gate
 * o_t = σ(W_o * [h_{t-1}, x_t] + b_o)     // Output gate
 * C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)  // Candidate cell state
 * C_t = f_t * C_{t-1} + i_t * C̃_t         // Cell state
 * h_t = o_t * tanh(C_t)                   // Hidden state
 */
public class LSTMCell {
    private final int inputSize;
    private final int hiddenSize;
    
    // Weight matrices for gates and cell state
    private double[][] W_f;  // Forget gate weights
    private double[][] W_i;  // Input gate weights
    private double[][] W_o;  // Output gate weights
    private double[][] W_C;  // Cell state weights
    
    // Bias vectors
    private double[] b_f;    // Forget gate bias
    private double[] b_i;    // Input gate bias
    private double[] b_o;    // Output gate bias
    private double[] b_C;    // Cell state bias
    
    // Current states
    private double[] hiddenState;
    private double[] cellState;
    
    /**
     * Forward pass for a single time step.
     * 
     * @param input Input vector for current time step
     * @return Hidden state for current time step
     */
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize);
        }
        
        // Concatenate input and previous hidden state
        double[] concat = new double[inputSize + hiddenSize];
        System.arraycopy(input, 0, concat, 0, inputSize);
        System.arraycopy(hiddenState, 0, concat, inputSize, hiddenSize);
        
        // Compute gates
        double[] f_t = computeGate(concat, W_f, b_f);  // Forget gate
        double[] i_t = computeGate(concat, W_i, b_i);  // Input gate
        double[] o_t = computeGate(concat, W_o, b_o);  // Output gate
        
        // Compute candidate cell state
        double[] C_tilde = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            C_tilde[i] = b_C[i];
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                C_tilde[i] += W_C[j][i] * concat[j];
            }
            C_tilde[i] = Math.tanh(C_tilde[i]);
        }
        
        // Update cell state: C_t = f_t * C_{t-1} + i_t * C̃_t
        double[] newCellState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            newCellState[i] = f_t[i] * cellState[i] + i_t[i] * C_tilde[i];
        }
        
        // Update hidden state: h_t = o_t * tanh(C_t)
        double[] newHiddenState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            newHiddenState[i] = o_t[i] * Math.tanh(newCellState[i]);
        }
        
        // Update current states
        hiddenState = newHiddenState;
        cellState = newCellState;
        
        return hiddenState.clone();
    }
}
```

## 9.3 Gated Recurrent Unit (GRU)

GRU is a simplified version of LSTM that uses only two gates instead of three, making it more computationally efficient while still addressing the vanishing gradient problem.

### 9.3.1 GRU Architecture

GRU uses two gates:
- **Update Gate**: `z_t = σ(W_z * [h_{t-1}, x_t] + b_z)`
- **Reset Gate**: `r_t = σ(W_r * [h_{t-1}, x_t] + b_r)`
- **Hidden State**: `h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)`

#### Implementation

```java
package com.aiprogramming.ch09.core;

/**
 * Gated Recurrent Unit (GRU) cell implementation.
 * 
 * GRU is a simplified version of LSTM that uses only two gates instead of three,
 * making it more computationally efficient while still addressing the vanishing
 * gradient problem.
 * 
 * Mathematical formulation:
 * z_t = σ(W_z * [h_{t-1}, x_t] + b_z)     // Update gate
 * r_t = σ(W_r * [h_{t-1}, x_t] + b_r)     // Reset gate
 * h̃_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)  // Candidate hidden state
 * h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t   // Hidden state
 */
public class GRUCell {
    private final int inputSize;
    private final int hiddenSize;
    
    // Weight matrices for gates and hidden state
    private double[][] W_z;  // Update gate weights
    private double[][] W_r;  // Reset gate weights
    private double[][] W_h;  // Hidden state weights
    
    // Bias vectors
    private double[] b_z;    // Update gate bias
    private double[] b_r;    // Reset gate bias
    private double[] b_h;    // Hidden state bias
    
    // Current hidden state
    private double[] hiddenState;
    
    /**
     * Forward pass for a single time step.
     * 
     * @param input Input vector for current time step
     * @return Hidden state for current time step
     */
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must be " + inputSize);
        }
        
        // Concatenate input and previous hidden state
        double[] concat = new double[inputSize + hiddenSize];
        System.arraycopy(input, 0, concat, 0, inputSize);
        System.arraycopy(hiddenState, 0, concat, inputSize, hiddenSize);
        
        // Compute update gate: z_t = σ(W_z * [h_{t-1}, x_t] + b_z)
        double[] z_t = computeGate(concat, W_z, b_z);
        
        // Compute reset gate: r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
        double[] r_t = computeGate(concat, W_r, b_r);
        
        // Compute candidate hidden state: h̃_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)
        double[] candidateConcat = new double[inputSize + hiddenSize];
        System.arraycopy(input, 0, candidateConcat, 0, inputSize);
        
        // Apply reset gate to previous hidden state
        for (int i = 0; i < hiddenSize; i++) {
            candidateConcat[inputSize + i] = r_t[i] * hiddenState[i];
        }
        
        double[] h_tilde = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            h_tilde[i] = b_h[i];
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                h_tilde[i] += W_h[j][i] * candidateConcat[j];
            }
            h_tilde[i] = Math.tanh(h_tilde[i]);
        }
        
        // Update hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
        double[] newHiddenState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            newHiddenState[i] = (1.0 - z_t[i]) * hiddenState[i] + z_t[i] * h_tilde[i];
        }
        
        // Update current hidden state
        hiddenState = newHiddenState;
        
        return hiddenState.clone();
    }
}
```

## 9.4 Applications of RNNs

### 9.4.1 Text Generation

RNNs can be used to generate text by learning patterns in character sequences.

#### Implementation

```java
package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.LSTMCell;
import com.aiprogramming.ch09.utils.Vocabulary;

/**
 * Text generation using Recurrent Neural Networks.
 * 
 * This class implements a character-level text generator that learns
 * patterns in text data and can generate new text based on a seed.
 */
public class TextGenerator {
    private LSTMCell lstm;
    private Vocabulary vocabulary;
    private double[][] outputWeights;
    private double[] outputBias;
    private int hiddenSize;
    private double learningRate;
    
    /**
     * Generate text based on a seed string.
     * 
     * @param seed Starting text
     * @param length Number of characters to generate
     * @return Generated text
     */
    public String generateText(String seed, int length) {
        if (lstm == null) {
            throw new IllegalStateException("Model must be trained before generating text");
        }
        
        lstm.resetState();
        StringBuilder result = new StringBuilder(seed);
        
        // Process seed characters
        String[] seedChars = seed.split("");
        for (String ch : seedChars) {
            if (vocabulary.contains(ch)) {
                double[] input = vocabulary.oneHotEncode(ch);
                lstm.forward(input);
            }
        }
        
        // Generate new characters
        for (int i = 0; i < length; i++) {
            double[] hiddenState = lstm.getHiddenState();
            double[] output = computeOutput(hiddenState);
            
            // Sample next character
            String nextChar = sampleCharacter(output);
            result.append(nextChar);
            
            // Feed back to LSTM
            if (vocabulary.contains(nextChar)) {
                double[] input = vocabulary.oneHotEncode(nextChar);
                lstm.forward(input);
            }
        }
        
        return result.toString();
    }
}
```

### 9.4.2 Sentiment Analysis

RNNs can analyze the sentiment of text by processing character or word sequences.

#### Implementation

```java
package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.GRUCell;
import com.aiprogramming.ch09.utils.Vocabulary;

/**
 * Sentiment analysis using Recurrent Neural Networks.
 * 
 * This class implements a sentiment analyzer that can classify text
 * as positive, negative, or neutral based on learned patterns.
 */
public class SentimentAnalyzer {
    private GRUCell gru;
    private Vocabulary vocabulary;
    private double[][] outputWeights;
    private double[] outputBias;
    private int hiddenSize;
    private double learningRate;
    private boolean isTrained;
    
    /**
     * Analyze sentiment of a given text.
     * 
     * @param text Input text to analyze
     * @return Sentiment score (0.0 = negative, 1.0 = positive)
     */
    public double analyzeSentiment(String text) {
        if (isTrained && gru != null) {
            return analyzeWithRNN(text);
        } else {
            return analyzeWithLexicon(text);
        }
    }
    
    /**
     * Analyze sentiment using the trained RNN model.
     * 
     * @param text Input text
     * @return Sentiment score
     */
    private double analyzeWithRNN(String text) {
        gru.resetState();
        
        // Process text character by character
        String[] characters = text.split("");
        for (String ch : characters) {
            if (vocabulary.contains(ch)) {
                double[] input = vocabulary.oneHotEncode(ch);
                gru.forward(input);
            }
        }
        
        // Get final hidden state and compute output
        double[] hiddenState = gru.getHiddenState();
        return computeOutput(hiddenState);
    }
}
```

### 9.4.3 Time Series Prediction

RNNs excel at time series forecasting by learning temporal patterns in sequential data.

#### Implementation

```java
package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.RNNCell;

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
}
```

## 9.5 Training Techniques

### 9.5.1 Gradient Clipping

Gradient clipping prevents exploding gradients by limiting the norm of gradients.

#### Implementation

```java
/**
 * Apply gradient clipping to prevent exploding gradients.
 * 
 * @param gradients List of gradient vectors
 * @param maxNorm Maximum allowed gradient norm
 * @return Clipped gradients
 */
public static List<double[]> clipGradients(List<double[]> gradients, double maxNorm) {
    // Compute total norm
    double totalNorm = 0.0;
    for (double[] grad : gradients) {
        for (double g : grad) {
            totalNorm += g * g;
        }
    }
    totalNorm = Math.sqrt(totalNorm);
    
    // Compute clip coefficient
    double clipCoeff = Math.min(1.0, maxNorm / totalNorm);
    
    // Apply clipping
    List<double[]> clippedGradients = new ArrayList<>();
    for (double[] grad : gradients) {
        double[] clipped = new double[grad.length];
        for (int i = 0; i < grad.length; i++) {
            clipped[i] = grad[i] * clipCoeff;
        }
        clippedGradients.add(clipped);
    }
    
    return clippedGradients;
}
```

### 9.5.2 Teacher Forcing

Teacher forcing is a training technique where the ground truth output is used as the next input during training.

#### Implementation

```java
/**
 * Train with teacher forcing.
 * 
 * @param inputSequence Input sequence
 * @param targetSequence Target sequence
 * @param useTeacherForcing Whether to use teacher forcing
 */
public void trainWithTeacherForcing(List<double[]> inputSequence, 
                                   List<double[]> targetSequence, 
                                   boolean useTeacherForcing) {
    resetState();
    
    for (int t = 0; t < inputSequence.size(); t++) {
        double[] input = inputSequence.get(t);
        
        if (useTeacherForcing && t > 0) {
            // Use target from previous time step as input
            input = targetSequence.get(t - 1);
        }
        
        double[] output = forward(input);
        // Compute loss and gradients...
    }
}
```

## 9.6 Advanced RNN Architectures

### 9.6.1 Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions.

### 9.6.2 Stacked RNNs

Stacked RNNs consist of multiple RNN layers stacked on top of each other.

### 9.6.3 Attention Mechanisms

Attention mechanisms allow RNNs to focus on relevant parts of the input sequence.

## 9.7 Practical Considerations

### 9.7.1 Choosing Between RNN, LSTM, and GRU

- **Simple RNN**: Good for short sequences, computationally efficient
- **LSTM**: Best for long sequences, handles vanishing gradients well
- **GRU**: Good balance between performance and efficiency

### 9.7.2 Hyperparameter Tuning

- **Hidden size**: Larger for complex patterns, smaller for efficiency
- **Learning rate**: Start with 0.01, adjust based on convergence
- **Sequence length**: Balance between context and computational cost

### 9.7.3 Data Preprocessing

- **Normalization**: Scale inputs to [0, 1] or [-1, 1] range
- **Padding**: Pad sequences to uniform length
- **Vocabulary**: Build vocabulary from training data

## 9.8 Summary

In this chapter, we explored Recurrent Neural Networks and their variants:

1. **Basic RNNs**: Simple recurrent networks with feedback connections
2. **LSTM**: Advanced RNN with cell state and three gates
3. **GRU**: Simplified LSTM with two gates
4. **Applications**: Text generation, sentiment analysis, time series prediction
5. **Training**: BPTT, gradient clipping, teacher forcing
6. **Architectures**: Bidirectional, stacked, attention mechanisms

RNNs are powerful tools for sequential data processing, enabling applications in natural language processing, speech recognition, and time series analysis. Understanding their architecture and training techniques is essential for building effective sequence models.

## Exercises

1. **Implement a simple RNN** from scratch and train it on a sequence prediction task
2. **Compare RNN, LSTM, and GRU** performance on a text generation task
3. **Build a sentiment analyzer** using RNNs and evaluate its performance
4. **Create a time series predictor** for stock price forecasting
5. **Experiment with gradient clipping** and observe its effect on training stability
6. **Implement teacher forcing** and compare training with and without it
7. **Build a character-level language model** and generate text samples
8. **Analyze the vanishing gradient problem** in simple RNNs vs LSTM/GRU

## Further Reading

- "Understanding LSTM Networks" by Christopher Olah
- "The Unreasonable Effectiveness of RNNs" by Andrej Karpathy
- "Long Short-Term Memory" by Hochreiter & Schmidhuber
- "Learning Phrase Representations using RNN Encoder-Decoder" by Cho et al.
- "Attention Is All You Need" by Vaswani et al.

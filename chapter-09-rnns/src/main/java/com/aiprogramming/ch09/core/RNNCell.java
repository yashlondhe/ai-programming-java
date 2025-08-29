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
    
    // Storage for backpropagation
    private List<double[]> hiddenStates;
    private List<double[]> inputs;
    private List<double[]> outputs;
    
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
     * Initialize weight matrices and bias vectors with random values.
     */
    private void initializeWeights() {
        Random random = new Random();
        
        // Initialize weight matrices with Xavier/Glorot initialization
        double inputScale = Math.sqrt(2.0 / (inputSize + hiddenSize));
        double hiddenScale = Math.sqrt(2.0 / (hiddenSize + hiddenSize));
        double outputScale = Math.sqrt(2.0 / (hiddenSize + outputSize));
        
        W_xh = new double[inputSize][hiddenSize];
        W_hh = new double[hiddenSize][hiddenSize];
        W_hy = new double[hiddenSize][outputSize];
        
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_xh[i][j] = random.nextGaussian() * inputScale;
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_hh[i][j] = random.nextGaussian() * hiddenScale;
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                W_hy[i][j] = random.nextGaussian() * outputScale;
            }
        }
        
        // Initialize bias vectors
        b_h = new double[hiddenSize];
        b_y = new double[outputSize];
        
        for (int i = 0; i < hiddenSize; i++) {
            b_h[i] = 0.0;
        }
        
        for (int i = 0; i < outputSize; i++) {
            b_y[i] = 0.0;
        }
    }
    
    /**
     * Reset the hidden state and clear stored states for new sequence.
     */
    public void resetState() {
        hiddenState = new double[hiddenSize];
        hiddenStates = new ArrayList<>();
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
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
        
        // Store input for backpropagation
        inputs.add(input.clone());
        
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
        hiddenStates.add(hiddenState.clone());
        
        // Compute output: y_t = W_hy * h_t + b_y
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = b_y[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += W_hy[j][i] * hiddenState[j];
            }
        }
        
        outputs.add(output.clone());
        return output;
    }
    
    /**
     * Forward pass for a sequence of inputs.
     * 
     * @param sequence List of input vectors
     * @return List of output vectors
     */
    public List<double[]> forwardSequence(List<double[]> sequence) {
        resetState();
        List<double[]> sequenceOutputs = new ArrayList<>();
        
        for (double[] input : sequence) {
            double[] output = forward(input);
            sequenceOutputs.add(output);
        }
        
        return sequenceOutputs;
    }
    
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
        
        // Return gradients with respect to inputs (for chaining)
        List<double[]> inputGradients = new ArrayList<>();
        for (int t = 0; t < inputs.size(); t++) {
            double[] dx = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dx[i] += dW_xh[i][j];
                }
            }
            inputGradients.add(dx);
        }
        
        return inputGradients;
    }
    
    /**
     * Update weights using computed gradients.
     * 
     * @param learningRate Learning rate for gradient descent
     */
    public void updateWeights(double learningRate) {
        if (dW_xh == null) {
            throw new IllegalStateException("Must call backward() before updateWeights()");
        }
        
        // Update weight matrices
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_xh[i][j] -= learningRate * dW_xh[i][j];
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_hh[i][j] -= learningRate * dW_hh[i][j];
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                W_hy[i][j] -= learningRate * dW_hy[i][j];
            }
        }
        
        // Update bias vectors
        for (int i = 0; i < hiddenSize; i++) {
            b_h[i] -= learningRate * db_h[i];
        }
        
        for (int i = 0; i < outputSize; i++) {
            b_y[i] -= learningRate * db_y[i];
        }
        
        // Clear gradients
        dW_xh = null;
        dW_hh = null;
        dW_hy = null;
        db_h = null;
        db_y = null;
    }
    
    /**
     * Get current hidden state.
     * 
     * @return Current hidden state vector
     */
    public double[] getHiddenState() {
        return hiddenState.clone();
    }
    
    /**
     * Set hidden state (useful for initialization).
     * 
     * @param hiddenState New hidden state
     */
    public void setHiddenState(double[] hiddenState) {
        if (hiddenState.length != this.hiddenSize) {
            throw new IllegalArgumentException("Hidden state size must be " + hiddenSize);
        }
        this.hiddenState = hiddenState.clone();
    }
    
    // Getters for testing and debugging
    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    public int getOutputSize() { return outputSize; }
    
    // Gradient storage for weight updates
    private double[][] dW_xh;
    private double[][] dW_hh;
    private double[][] dW_hy;
    private double[] db_h;
    private double[] db_y;
}

package com.aiprogramming.ch09.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
    
    // Storage for backpropagation
    private List<double[]> hiddenStates;
    private List<double[]> inputs;
    private List<double[]> updateGates;
    private List<double[]> resetGates;
    private List<double[]> candidateHiddenStates;
    
    /**
     * Constructor for GRU cell.
     * 
     * @param inputSize Size of input vectors
     * @param hiddenSize Size of hidden state
     */
    public GRUCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        
        initializeWeights();
        resetState();
    }
    
    /**
     * Initialize weight matrices and bias vectors with random values.
     */
    private void initializeWeights() {
        Random random = new Random();
        
        // Initialize weight matrices with Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
        
        W_z = new double[inputSize + hiddenSize][hiddenSize];
        W_r = new double[inputSize + hiddenSize][hiddenSize];
        W_h = new double[inputSize + hiddenSize][hiddenSize];
        
        for (int i = 0; i < inputSize + hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_z[i][j] = random.nextGaussian() * scale;
                W_r[i][j] = random.nextGaussian() * scale;
                W_h[i][j] = random.nextGaussian() * scale;
            }
        }
        
        // Initialize bias vectors
        b_z = new double[hiddenSize];
        b_r = new double[hiddenSize];
        b_h = new double[hiddenSize];
        
        for (int i = 0; i < hiddenSize; i++) {
            b_z[i] = 0.0;
            b_r[i] = 0.0;
            b_h[i] = 0.0;
        }
    }
    
    /**
     * Reset the hidden state and clear stored states for new sequence.
     */
    public void resetState() {
        hiddenState = new double[hiddenSize];
        hiddenStates = new ArrayList<>();
        inputs = new ArrayList<>();
        updateGates = new ArrayList<>();
        resetGates = new ArrayList<>();
        candidateHiddenStates = new ArrayList<>();
    }
    
    /**
     * Sigmoid activation function.
     * 
     * @param x Input value
     * @return Sigmoid output
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Apply sigmoid activation to a vector.
     * 
     * @param x Input vector
     * @return Sigmoid output vector
     */
    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = sigmoid(x[i]);
        }
        return result;
    }
    
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
        
        // Store input for backpropagation
        inputs.add(input.clone());
        
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
        
        // Store states for backpropagation
        hiddenStates.add(hiddenState.clone());
        updateGates.add(z_t);
        resetGates.add(r_t);
        candidateHiddenStates.add(h_tilde);
        
        // Update current hidden state
        hiddenState = newHiddenState;
        
        return hiddenState.clone();
    }
    
    /**
     * Forward pass for a sequence of inputs.
     * 
     * @param sequence List of input vectors
     * @return List of hidden state vectors
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
     * Compute gate values using sigmoid activation.
     * 
     * @param concat Concatenated input and hidden state
     * @param W Weight matrix for the gate
     * @param b Bias vector for the gate
     * @return Gate values
     */
    private double[] computeGate(double[] concat, double[][] W, double[] b) {
        double[] gate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            gate[i] = b[i];
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                gate[i] += W[j][i] * concat[j];
            }
        }
        return sigmoid(gate);
    }
    
    /**
     * Backward pass through time (BPTT) for computing gradients.
     * 
     * @param hiddenGradients Gradients with respect to hidden states
     * @return Gradients with respect to inputs
     */
    public List<double[]> backward(List<double[]> hiddenGradients) {
        if (hiddenGradients.size() != inputs.size()) {
            throw new IllegalArgumentException("Hidden gradients size must match inputs size");
        }
        
        // Initialize gradient accumulators
        double[][] dW_z = new double[inputSize + hiddenSize][hiddenSize];
        double[][] dW_r = new double[inputSize + hiddenSize][hiddenSize];
        double[][] dW_h = new double[inputSize + hiddenSize][hiddenSize];
        double[] db_z = new double[hiddenSize];
        double[] db_r = new double[hiddenSize];
        double[] db_h = new double[hiddenSize];
        
        // Initialize gradient for next time step
        double[] dh_next = new double[hiddenSize];
        
        // Backward pass through time
        for (int t = inputs.size() - 1; t >= 0; t--) {
            // Gradient with respect to hidden state
            double[] dh = hiddenGradients.get(t).clone();
            for (int i = 0; i < hiddenSize; i++) {
                dh[i] += dh_next[i];
            }
            
            // Gradient with respect to update gate
            double[] dz_t = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dz_t[i] = dh[i] * (candidateHiddenStates.get(t)[i] - hiddenStates.get(t)[i]) * 
                          updateGates.get(t)[i] * (1.0 - updateGates.get(t)[i]);
            }
            
            // Gradient with respect to candidate hidden state
            double[] dh_tilde = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dh_tilde[i] = dh[i] * updateGates.get(t)[i] * 
                              (1.0 - Math.pow(candidateHiddenStates.get(t)[i], 2));
            }
            
            // Gradient with respect to reset gate
            double[] dr_t = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                double sum = 0.0;
                for (int j = 0; j < hiddenSize; j++) {
                    sum += W_h[inputSize + i][j] * dh_tilde[j];
                }
                dr_t[i] = sum * hiddenStates.get(t)[i] * 
                          resetGates.get(t)[i] * (1.0 - resetGates.get(t)[i]);
            }
            
            // Concatenate input and previous hidden state
            double[] concat = new double[inputSize + hiddenSize];
            System.arraycopy(inputs.get(t), 0, concat, 0, inputSize);
            System.arraycopy(hiddenStates.get(t), 0, concat, inputSize, hiddenSize);
            
            // Candidate concatenation for hidden state computation
            double[] candidateConcat = new double[inputSize + hiddenSize];
            System.arraycopy(inputs.get(t), 0, candidateConcat, 0, inputSize);
            for (int i = 0; i < hiddenSize; i++) {
                candidateConcat[inputSize + i] = resetGates.get(t)[i] * hiddenStates.get(t)[i];
            }
            
            // Accumulate gradients for weights and biases
            for (int i = 0; i < inputSize + hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dW_z[i][j] += concat[i] * dz_t[j];
                    dW_r[i][j] += concat[i] * dr_t[j];
                    dW_h[i][j] += candidateConcat[i] * dh_tilde[j];
                }
            }
            
            for (int j = 0; j < hiddenSize; j++) {
                db_z[j] += dz_t[j];
                db_r[j] += dr_t[j];
                db_h[j] += dh_tilde[j];
            }
            
            // Compute gradient for next time step
            if (t > 0) {
                dh_next = new double[hiddenSize];
                
                for (int i = 0; i < hiddenSize; i++) {
                    // Gradient from update gate
                    dh_next[i] += dh[i] * (1.0 - updateGates.get(t)[i]);
                    
                    // Gradient from reset gate
                    for (int j = 0; j < hiddenSize; j++) {
                        dh_next[i] += W_r[inputSize + i][j] * dr_t[j];
                    }
                    
                    // Gradient from candidate hidden state
                    for (int j = 0; j < hiddenSize; j++) {
                        dh_next[i] += W_h[inputSize + i][j] * dh_tilde[j] * resetGates.get(t)[i];
                    }
                }
            }
        }
        
        // Store gradients for weight updates
        this.dW_z = dW_z;
        this.dW_r = dW_r;
        this.dW_h = dW_h;
        this.db_z = db_z;
        this.db_r = db_r;
        this.db_h = db_h;
        
        // Return gradients with respect to inputs (for chaining)
        List<double[]> inputGradients = new ArrayList<>();
        for (int t = 0; t < inputs.size(); t++) {
            double[] dx = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dx[i] += dW_z[i][j] + dW_r[i][j] + dW_h[i][j];
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
        if (dW_z == null) {
            throw new IllegalStateException("Must call backward() before updateWeights()");
        }
        
        // Update weight matrices
        for (int i = 0; i < inputSize + hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_z[i][j] -= learningRate * dW_z[i][j];
                W_r[i][j] -= learningRate * dW_r[i][j];
                W_h[i][j] -= learningRate * dW_h[i][j];
            }
        }
        
        // Update bias vectors
        for (int i = 0; i < hiddenSize; i++) {
            b_z[i] -= learningRate * db_z[i];
            b_r[i] -= learningRate * db_r[i];
            b_h[i] -= learningRate * db_h[i];
        }
        
        // Clear gradients
        dW_z = null;
        dW_r = null;
        dW_h = null;
        db_z = null;
        db_r = null;
        db_h = null;
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
    
    // Gradient storage for weight updates
    private double[][] dW_z;
    private double[][] dW_r;
    private double[][] dW_h;
    private double[] db_z;
    private double[] db_r;
    private double[] db_h;
}

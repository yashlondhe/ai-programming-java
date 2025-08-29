package com.aiprogramming.ch09.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
    
    // Storage for backpropagation
    private List<double[]> hiddenStates;
    private List<double[]> cellStates;
    private List<double[]> inputs;
    private List<double[]> forgetGates;
    private List<double[]> inputGates;
    private List<double[]> outputGates;
    private List<double[]> candidateCellStates;
    
    /**
     * Constructor for LSTM cell.
     * 
     * @param inputSize Size of input vectors
     * @param hiddenSize Size of hidden state
     */
    public LSTMCell(int inputSize, int hiddenSize) {
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
        
        W_f = new double[inputSize + hiddenSize][hiddenSize];
        W_i = new double[inputSize + hiddenSize][hiddenSize];
        W_o = new double[inputSize + hiddenSize][hiddenSize];
        W_C = new double[inputSize + hiddenSize][hiddenSize];
        
        for (int i = 0; i < inputSize + hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_f[i][j] = random.nextGaussian() * scale;
                W_i[i][j] = random.nextGaussian() * scale;
                W_o[i][j] = random.nextGaussian() * scale;
                W_C[i][j] = random.nextGaussian() * scale;
            }
        }
        
        // Initialize bias vectors
        b_f = new double[hiddenSize];
        b_i = new double[hiddenSize];
        b_o = new double[hiddenSize];
        b_C = new double[hiddenSize];
        
        // Initialize forget gate bias to 1.0 to help with gradient flow
        for (int i = 0; i < hiddenSize; i++) {
            b_f[i] = 1.0;
            b_i[i] = 0.0;
            b_o[i] = 0.0;
            b_C[i] = 0.0;
        }
    }
    
    /**
     * Reset the hidden state, cell state, and clear stored states for new sequence.
     */
    public void resetState() {
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];
        hiddenStates = new ArrayList<>();
        cellStates = new ArrayList<>();
        inputs = new ArrayList<>();
        forgetGates = new ArrayList<>();
        inputGates = new ArrayList<>();
        outputGates = new ArrayList<>();
        candidateCellStates = new ArrayList<>();
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
        
        // Store states for backpropagation
        hiddenStates.add(hiddenState.clone());
        cellStates.add(cellState.clone());
        forgetGates.add(f_t);
        inputGates.add(i_t);
        outputGates.add(o_t);
        candidateCellStates.add(C_tilde);
        
        // Update current states
        hiddenState = newHiddenState;
        cellState = newCellState;
        
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
        double[][] dW_f = new double[inputSize + hiddenSize][hiddenSize];
        double[][] dW_i = new double[inputSize + hiddenSize][hiddenSize];
        double[][] dW_o = new double[inputSize + hiddenSize][hiddenSize];
        double[][] dW_C = new double[inputSize + hiddenSize][hiddenSize];
        double[] db_f = new double[hiddenSize];
        double[] db_i = new double[hiddenSize];
        double[] db_o = new double[hiddenSize];
        double[] db_C = new double[hiddenSize];
        
        // Initialize gradients for next time step
        double[] dh_next = new double[hiddenSize];
        double[] dC_next = new double[hiddenSize];
        
        // Backward pass through time
        for (int t = inputs.size() - 1; t >= 0; t--) {
            // Gradient with respect to hidden state
            double[] dh = hiddenGradients.get(t).clone();
            for (int i = 0; i < hiddenSize; i++) {
                dh[i] += dh_next[i];
            }
            
            // Gradient with respect to cell state
            double[] dC = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dC[i] = dh[i] * outputGates.get(t)[i] * (1.0 - Math.pow(Math.tanh(cellStates.get(t + 1)[i]), 2));
                dC[i] += dC_next[i];
            }
            
            // Gradient with respect to output gate
            double[] do_t = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                do_t[i] = dh[i] * Math.tanh(cellStates.get(t + 1)[i]) * 
                          outputGates.get(t)[i] * (1.0 - outputGates.get(t)[i]);
            }
            
            // Gradient with respect to cell state (candidate)
            double[] dC_tilde = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dC_tilde[i] = dC[i] * inputGates.get(t)[i] * 
                              (1.0 - Math.pow(candidateCellStates.get(t)[i], 2));
            }
            
            // Gradient with respect to input gate
            double[] di_t = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                di_t[i] = dC[i] * candidateCellStates.get(t)[i] * 
                          inputGates.get(t)[i] * (1.0 - inputGates.get(t)[i]);
            }
            
            // Gradient with respect to forget gate
            double[] df_t = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                df_t[i] = dC[i] * (t > 0 ? cellStates.get(t)[i] : 0.0) * 
                          forgetGates.get(t)[i] * (1.0 - forgetGates.get(t)[i]);
            }
            
            // Concatenate input and previous hidden state
            double[] concat = new double[inputSize + hiddenSize];
            System.arraycopy(inputs.get(t), 0, concat, 0, inputSize);
            if (t > 0) {
                System.arraycopy(hiddenStates.get(t), 0, concat, inputSize, hiddenSize);
            }
            
            // Accumulate gradients for weights and biases
            for (int i = 0; i < inputSize + hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dW_f[i][j] += concat[i] * df_t[j];
                    dW_i[i][j] += concat[i] * di_t[j];
                    dW_o[i][j] += concat[i] * do_t[j];
                    dW_C[i][j] += concat[i] * dC_tilde[j];
                }
            }
            
            for (int j = 0; j < hiddenSize; j++) {
                db_f[j] += df_t[j];
                db_i[j] += di_t[j];
                db_o[j] += do_t[j];
                db_C[j] += dC_tilde[j];
            }
            
            // Compute gradients for next time step
            if (t > 0) {
                dh_next = new double[hiddenSize];
                dC_next = new double[hiddenSize];
                
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        dh_next[i] += W_f[inputSize + i][j] * df_t[j];
                        dh_next[i] += W_i[inputSize + i][j] * di_t[j];
                        dh_next[i] += W_o[inputSize + i][j] * do_t[j];
                        dh_next[i] += W_C[inputSize + i][j] * dC_tilde[j];
                    }
                    dC_next[i] = dC[i] * forgetGates.get(t)[i];
                }
            }
        }
        
        // Store gradients for weight updates
        this.dW_f = dW_f;
        this.dW_i = dW_i;
        this.dW_o = dW_o;
        this.dW_C = dW_C;
        this.db_f = db_f;
        this.db_i = db_i;
        this.db_o = db_o;
        this.db_C = db_C;
        
        // Return gradients with respect to inputs (for chaining)
        List<double[]> inputGradients = new ArrayList<>();
        for (int t = 0; t < inputs.size(); t++) {
            double[] dx = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dx[i] += dW_f[i][j] + dW_i[i][j] + dW_o[i][j] + dW_C[i][j];
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
        if (dW_f == null) {
            throw new IllegalStateException("Must call backward() before updateWeights()");
        }
        
        // Update weight matrices
        for (int i = 0; i < inputSize + hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W_f[i][j] -= learningRate * dW_f[i][j];
                W_i[i][j] -= learningRate * dW_i[i][j];
                W_o[i][j] -= learningRate * dW_o[i][j];
                W_C[i][j] -= learningRate * dW_C[i][j];
            }
        }
        
        // Update bias vectors
        for (int i = 0; i < hiddenSize; i++) {
            b_f[i] -= learningRate * db_f[i];
            b_i[i] -= learningRate * db_i[i];
            b_o[i] -= learningRate * db_o[i];
            b_C[i] -= learningRate * db_C[i];
        }
        
        // Clear gradients
        dW_f = null;
        dW_i = null;
        dW_o = null;
        dW_C = null;
        db_f = null;
        db_i = null;
        db_o = null;
        db_C = null;
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
     * Get current cell state.
     * 
     * @return Current cell state vector
     */
    public double[] getCellState() {
        return cellState.clone();
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
    
    /**
     * Set cell state (useful for initialization).
     * 
     * @param cellState New cell state
     */
    public void setCellState(double[] cellState) {
        if (cellState.length != this.hiddenSize) {
            throw new IllegalArgumentException("Cell state size must be " + hiddenSize);
        }
        this.cellState = cellState.clone();
    }
    
    // Getters for testing and debugging
    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    
    // Gradient storage for weight updates
    private double[][] dW_f;
    private double[][] dW_i;
    private double[][] dW_o;
    private double[][] dW_C;
    private double[] db_f;
    private double[] db_i;
    private double[] db_o;
    private double[] db_C;
}

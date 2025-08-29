package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Artificial neuron implementation
 */
public class Neuron {
    private List<Double> weights;
    private double bias;
    private ActivationFunction activationFunction;
    private double lastOutput;
    private List<Double> lastInputs;
    private List<Double> lastWeightGradients;
    private double lastBiasGradient;
    
    public Neuron(int inputSize, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.weights = new ArrayList<>();
        this.bias = 0.0;
        
        // Initialize weights randomly
        Random random = new Random();
        for (int i = 0; i < inputSize; i++) {
            weights.add(random.nextGaussian() * 0.1); // Small random weights
        }
    }
    
    /**
     * Forward pass: compute output given inputs
     */
    public double forward(List<Double> inputs) {
        if (inputs.size() != weights.size()) {
            throw new IllegalArgumentException("Input size must match weight size");
        }
        
        this.lastInputs = new ArrayList<>(inputs);
        
        // Compute weighted sum
        double weightedSum = bias;
        for (int i = 0; i < inputs.size(); i++) {
            weightedSum += weights.get(i) * inputs.get(i);
        }
        
        // Apply activation function
        this.lastOutput = activationFunction.apply(weightedSum);
        return this.lastOutput;
    }
    
    /**
     * Compute gradients for backpropagation
     */
    public List<Double> computeGradients(double outputGradient) {
        // Compute activation gradient
        double activationGradient = activationFunction.derivative(lastOutput) * outputGradient;
        
        // Compute weight gradients
        List<Double> weightGradients = new ArrayList<>();
        for (Double input : lastInputs) {
            weightGradients.add(input * activationGradient);
        }
        
        // Compute bias gradient
        double biasGradient = activationGradient;
        
        // Store gradients for weight updates
        this.lastWeightGradients = weightGradients;
        this.lastBiasGradient = biasGradient;
        
        // Return input gradients for backpropagation
        List<Double> inputGradients = new ArrayList<>();
        for (Double weight : weights) {
            inputGradients.add(weight * activationGradient);
        }
        
        return inputGradients;
    }
    
    /**
     * Update weights and bias using computed gradients
     */
    public void updateWeights(double learningRate) {
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * lastWeightGradients.get(i));
        }
        bias -= learningRate * lastBiasGradient;
    }
    
    // Getters
    public List<Double> getWeights() {
        return new ArrayList<>(weights);
    }
    
    public double getBias() {
        return bias;
    }
    
    public double getLastOutput() {
        return lastOutput;
    }
}

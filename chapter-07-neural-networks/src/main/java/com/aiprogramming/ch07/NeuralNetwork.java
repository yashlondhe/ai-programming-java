package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;

/**
 * Feedforward neural network implementation
 */
public class NeuralNetwork {
    private List<Layer> layers;
    private LossFunction lossFunction;
    private Optimizer optimizer;
    private boolean trained;
    
    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.lossFunction = new MeanSquaredError();
        this.optimizer = new SGD(0.01);
        this.trained = false;
    }
    
    /**
     * Add a layer to the network
     */
    public void addLayer(Layer layer) {
        layers.add(layer);
    }
    
    /**
     * Set loss function
     */
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    
    /**
     * Set optimizer
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }
    
    /**
     * Forward pass through the network
     */
    public List<Double> forward(List<Double> inputs) {
        List<Double> currentInputs = new ArrayList<>(inputs);
        
        for (Layer layer : layers) {
            currentInputs = layer.forward(currentInputs);
        }
        
        return currentInputs;
    }
    
    /**
     * Backward pass through the network
     */
    public void backward(List<Double> targets) {
        // Compute output layer gradients
        Layer outputLayer = layers.get(layers.size() - 1);
        List<Double> outputGradients = lossFunction.derivative(outputLayer.getLastOutputs(), targets);
        
        // Backpropagate through layers
        List<Double> currentGradients = outputGradients;
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentGradients = layers.get(i).backward(currentGradients);
        }
    }
    
    /**
     * Update weights using the optimizer
     */
    public void updateWeights() {
        for (Layer layer : layers) {
            layer.updateWeights(optimizer.getLearningRate());
        }
    }
    
    /**
     * Train the network for one epoch
     */
    public double trainEpoch(List<List<Double>> inputs, List<List<Double>> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Input and target sizes must match");
        }
        
        double totalLoss = 0.0;
        
        for (int i = 0; i < inputs.size(); i++) {
            // Forward pass
            List<Double> outputs = forward(inputs.get(i));
            
            // Compute loss
            double loss = lossFunction.compute(outputs, targets.get(i));
            totalLoss += loss;
            
            // Backward pass
            backward(targets.get(i));
            
            // Update weights
            updateWeights();
        }
        
        this.trained = true;
        return totalLoss / inputs.size();
    }
    
    /**
     * Train the network for multiple epochs
     */
    public List<Double> train(List<List<Double>> inputs, List<List<Double>> targets, int epochs) {
        List<Double> losses = new ArrayList<>();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = trainEpoch(inputs, targets);
            losses.add(loss);
            
            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, loss);
            }
        }
        
        return losses;
    }
    
    /**
     * Predict outputs for given inputs
     */
    public List<Double> predict(List<Double> inputs) {
        return forward(inputs);
    }
    
    /**
     * Predict outputs for multiple inputs
     */
    public List<List<Double>> predictBatch(List<List<Double>> inputs) {
        List<List<Double>> predictions = new ArrayList<>();
        for (List<Double> input : inputs) {
            predictions.add(predict(input));
        }
        return predictions;
    }
    
    /**
     * Get loss function for evaluation
     */
    public LossFunction getLossFunction() {
        return lossFunction;
    }
}

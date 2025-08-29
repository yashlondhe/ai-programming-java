package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Abstract layer class
 */
public abstract class Layer {
    protected List<Double> lastInputs;
    protected List<Double> lastOutputs;
    
    public abstract List<Double> forward(List<Double> inputs);
    public abstract List<Double> backward(List<Double> outputGradients);
    public abstract void updateWeights(double learningRate);
    
    public List<Double> getLastOutputs() {
        return new ArrayList<>(lastOutputs);
    }
}

/**
 * Dense (fully connected) layer
 */
class DenseLayer extends Layer {
    private List<Neuron> neurons;
    private List<List<Double>> lastWeightGradients;
    private List<Double> lastBiasGradients;
    
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < outputSize; i++) {
            neurons.add(new Neuron(inputSize, activationFunction));
        }
    }
    
    @Override
    public List<Double> forward(List<Double> inputs) {
        this.lastInputs = new ArrayList<>(inputs);
        this.lastOutputs = new ArrayList<>();
        
        for (Neuron neuron : neurons) {
            lastOutputs.add(neuron.forward(inputs));
        }
        
        return new ArrayList<>(lastOutputs);
    }
    
    @Override
    public List<Double> backward(List<Double> outputGradients) {
        if (outputGradients.size() != neurons.size()) {
            throw new IllegalArgumentException("Output gradients size must match neuron count");
        }
        
        // Compute gradients for each neuron
        this.lastWeightGradients = new ArrayList<>();
        this.lastBiasGradients = new ArrayList<>();
        
        List<Double> inputGradients = new ArrayList<>();
        for (int i = 0; i < lastInputs.size(); i++) {
            inputGradients.add(0.0);
        }
        
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            List<Double> neuronGradients = neuron.computeGradients(outputGradients.get(i));
            
            // Accumulate input gradients
            for (int j = 0; j < neuronGradients.size(); j++) {
                inputGradients.set(j, inputGradients.get(j) + neuronGradients.get(j));
            }
        }
        
        return inputGradients;
    }
    
    @Override
    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }
    
    public List<Neuron> getNeurons() {
        return neurons;
    }
}

/**
 * Softmax layer for multi-class classification
 */
class SoftmaxLayer extends Layer {
    
    @Override
    public List<Double> forward(List<Double> inputs) {
        this.lastInputs = new ArrayList<>(inputs);
        
        // Compute softmax: exp(x_i) / sum(exp(x_j))
        double maxInput = inputs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        
        List<Double> expInputs = inputs.stream()
                .map(input -> Math.exp(input - maxInput)) // Subtract max for numerical stability
                .collect(Collectors.toList());
        
        double sumExp = expInputs.stream().mapToDouble(Double::doubleValue).sum();
        
        this.lastOutputs = expInputs.stream()
                .map(exp -> exp / sumExp)
                .collect(Collectors.toList());
        
        return new ArrayList<>(this.lastOutputs);
    }
    
    @Override
    public List<Double> backward(List<Double> outputGradients) {
        // For softmax with cross-entropy loss, the gradient is simplified
        // The derivative of softmax + cross-entropy is: softmax_i - target_i
        return new ArrayList<>(outputGradients);
    }
    
    @Override
    public void updateWeights(double learningRate) {
        // Softmax layer has no weights to update
    }
}

/**
 * Dropout layer for regularization
 */
class DropoutLayer extends Layer {
    private double dropoutRate;
    private List<Boolean> mask;
    private boolean isTraining;
    
    public DropoutLayer(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        this.isTraining = true;
    }
    
    public void setTraining(boolean training) {
        this.isTraining = training;
    }
    
    @Override
    public List<Double> forward(List<Double> inputs) {
        this.lastInputs = new ArrayList<>(inputs);
        this.lastOutputs = new ArrayList<>();
        
        if (isTraining) {
            // Create dropout mask
            Random random = new Random();
            this.mask = new ArrayList<>();
            for (int i = 0; i < inputs.size(); i++) {
                mask.add(random.nextDouble() > dropoutRate);
            }
            
            // Apply mask
            for (int i = 0; i < inputs.size(); i++) {
                lastOutputs.add(mask.get(i) ? inputs.get(i) / (1.0 - dropoutRate) : 0.0);
            }
        } else {
            // During inference, no dropout
            lastOutputs.addAll(inputs);
        }
        
        return new ArrayList<>(lastOutputs);
    }
    
    @Override
    public List<Double> backward(List<Double> outputGradients) {
        List<Double> inputGradients = new ArrayList<>();
        
        for (int i = 0; i < outputGradients.size(); i++) {
            if (isTraining && mask.get(i)) {
                inputGradients.add(outputGradients.get(i) / (1.0 - dropoutRate));
            } else if (isTraining) {
                inputGradients.add(0.0);
            } else {
                inputGradients.add(outputGradients.get(i));
            }
        }
        
        return inputGradients;
    }
    
    @Override
    public void updateWeights(double learningRate) {
        // Dropout layer has no weights to update
    }
}

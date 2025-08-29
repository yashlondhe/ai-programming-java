package com.aiprogramming.ch08;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Convolutional Neural Network implementation
 */
public class CNN {
    private List<CNNLayer> layers;
    private List<DenseLayer> denseLayers;
    private double learningRate;
    private boolean trained;
    
    public CNN(double learningRate) {
        this.layers = new ArrayList<>();
        this.denseLayers = new ArrayList<>();
        this.learningRate = learningRate;
        this.trained = false;
    }
    
    /**
     * Add a convolutional layer
     */
    public void addConvLayer(int numFilters, int kernelSize, int inputChannels, int stride, String padding) {
        layers.add(new Conv2DLayer(numFilters, kernelSize, inputChannels, stride, padding));
    }
    
    /**
     * Add a max pooling layer
     */
    public void addMaxPoolLayer(int poolSize, int stride) {
        layers.add(new MaxPoolingLayer(poolSize, stride));
    }
    
    /**
     * Add an average pooling layer
     */
    public void addAveragePoolLayer(int poolSize, int stride) {
        layers.add(new AveragePoolingLayer(poolSize, stride));
    }
    
    /**
     * Add a global average pooling layer
     */
    public void addGlobalAveragePoolLayer() {
        layers.add(new GlobalAveragePoolingLayer());
    }
    
    /**
     * Add a dense (fully connected) layer
     */
    public void addDenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        denseLayers.add(new DenseLayer(inputSize, outputSize, activationFunction));
    }
    
    /**
     * Forward pass through the network
     */
    public Tensor forward(Tensor input) {
        Tensor currentInput = input;
        
        // Pass through CNN layers
        for (CNNLayer layer : layers) {
            currentInput = layer.forward(currentInput);
        }
        
        // Flatten the output for dense layers
        Tensor flattened = flatten(currentInput);
        
        // Pass through dense layers
        for (DenseLayer layer : denseLayers) {
            flattened = layer.forward(flattened);
        }
        
        return flattened;
    }
    
    /**
     * Backward pass through the network
     */
    public void backward(Tensor outputGradient) {
        Tensor currentGradient = outputGradient;
        
        // Backpropagate through dense layers
        for (int i = denseLayers.size() - 1; i >= 0; i--) {
            currentGradient = denseLayers.get(i).backward(currentGradient);
        }
        
        // Reshape gradient for CNN layers
        Tensor reshapedGradient = reshapeForCNN(currentGradient, layers.get(layers.size() - 1).getLastOutput().getShape());
        
        // Backpropagate through CNN layers
        for (int i = layers.size() - 1; i >= 0; i--) {
            reshapedGradient = layers.get(i).backward(reshapedGradient);
        }
    }
    
    /**
     * Update weights using computed gradients
     */
    public void updateWeights() {
        for (CNNLayer layer : layers) {
            layer.updateWeights(learningRate);
        }
        
        for (DenseLayer layer : denseLayers) {
            layer.updateWeights(learningRate);
        }
    }
    
    /**
     * Train the network for one epoch
     */
    public double trainEpoch(List<Tensor> inputs, List<Tensor> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Input and target sizes must match");
        }
        
        double totalLoss = 0.0;
        
        for (int i = 0; i < inputs.size(); i++) {
            // Forward pass
            Tensor output = forward(inputs.get(i));
            
            // Compute loss
            double loss = computeLoss(output, targets.get(i));
            totalLoss += loss;
            
            // Compute output gradient
            Tensor outputGradient = computeLossGradient(output, targets.get(i));
            
            // Backward pass
            backward(outputGradient);
            
            // Update weights
            updateWeights();
        }
        
        this.trained = true;
        return totalLoss / inputs.size();
    }
    
    /**
     * Train the network for multiple epochs
     */
    public List<Double> train(List<Tensor> inputs, List<Tensor> targets, int epochs) {
        List<Double> losses = new ArrayList<>();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = trainEpoch(inputs, targets);
            losses.add(loss);
            
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, loss);
            }
        }
        
        return losses;
    }
    
    /**
     * Predict output for given input
     */
    public Tensor predict(Tensor input) {
        return forward(input);
    }
    
    /**
     * Predict outputs for multiple inputs
     */
    public List<Tensor> predict(List<Tensor> inputs) {
        List<Tensor> predictions = new ArrayList<>();
        for (Tensor input : inputs) {
            predictions.add(predict(input));
        }
        return predictions;
    }
    
    /**
     * Flatten a 3D tensor to 1D for dense layers
     */
    private Tensor flatten(Tensor tensor) {
        int[] shape = tensor.getShape();
        int totalSize = tensor.getTotalSize();
        
        // Reshape to 1D
        double[] data = tensor.getData();
        return new Tensor(data, totalSize);
    }
    
    /**
     * Reshape 1D tensor back to 3D for CNN layers
     */
    private Tensor reshapeForCNN(Tensor tensor, int[] targetShape) {
        double[] data = tensor.getData();
        return new Tensor(data, targetShape);
    }
    
    /**
     * Compute loss between prediction and target
     */
    private double computeLoss(Tensor prediction, Tensor target) {
        // Mean squared error
        double[] predData = prediction.getData();
        double[] targetData = target.getData();
        
        double sum = 0.0;
        for (int i = 0; i < predData.length; i++) {
            double diff = predData[i] - targetData[i];
            sum += diff * diff;
        }
        
        return sum / predData.length;
    }
    
    /**
     * Compute gradient of loss with respect to prediction
     */
    private Tensor computeLossGradient(Tensor prediction, Tensor target) {
        // Gradient of mean squared error
        double[] predData = prediction.getData();
        double[] targetData = target.getData();
        
        double[] gradientData = new double[predData.length];
        for (int i = 0; i < predData.length; i++) {
            gradientData[i] = 2.0 * (predData[i] - targetData[i]) / predData.length;
        }
        
        return new Tensor(gradientData, predData.length);
    }
    
    /**
     * Set learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Get learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Check if network is trained
     */
    public boolean isTrained() {
        return trained;
    }
}

/**
 * Dense layer for the fully connected part of CNN
 */
class DenseLayer {
    private List<Neuron> neurons;
    private List<List<Double>> lastWeightGradients;
    private List<Double> lastBiasGradients;
    
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i < outputSize; i++) {
            neurons.add(new Neuron(inputSize, activationFunction));
        }
    }
    
    public Tensor forward(Tensor input) {
        // Convert tensor to list
        List<Double> inputList = new ArrayList<>();
        double[] inputData = input.getData();
        for (double val : inputData) {
            inputList.add(val);
        }
        
        // Process through neurons
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.forward(inputList));
        }
        
        // Convert back to tensor
        double[] outputData = new double[outputs.size()];
        for (int i = 0; i < outputs.size(); i++) {
            outputData[i] = outputs.get(i);
        }
        
        return new Tensor(outputData, outputs.size());
    }
    
    public Tensor backward(Tensor outputGradient) {
        // Convert tensor to list
        List<Double> outputGradList = new ArrayList<>();
        double[] outputGradData = outputGradient.getData();
        for (double val : outputGradData) {
            outputGradList.add(val);
        }
        
        // Compute gradients for each neuron
        List<Double> inputGradients = new ArrayList<>();
        for (int i = 0; i < neurons.get(0).getWeights().size(); i++) {
            inputGradients.add(0.0);
        }
        
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            List<Double> neuronGradients = neuron.computeGradients(outputGradList.get(i));
            
            // Accumulate input gradients
            for (int j = 0; j < neuronGradients.size(); j++) {
                inputGradients.set(j, inputGradients.get(j) + neuronGradients.get(j));
            }
        }
        
        // Convert back to tensor
        double[] inputGradData = new double[inputGradients.size()];
        for (int i = 0; i < inputGradients.size(); i++) {
            inputGradData[i] = inputGradients.get(i);
        }
        
        return new Tensor(inputGradData, inputGradients.size());
    }
    
    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }
}

/**
 * Neuron class for dense layers
 */
class Neuron {
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
            weights.add(random.nextGaussian() * 0.1);
        }
    }
    
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
    
    public void updateWeights(double learningRate) {
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i) - learningRate * lastWeightGradients.get(i));
        }
        bias -= learningRate * lastBiasGradient;
    }
    
    public List<Double> getWeights() {
        return new ArrayList<>(weights);
    }
}

/**
 * Activation function interface
 */
interface ActivationFunction {
    double apply(double x);
    double derivative(double x);
}

/**
 * ReLU activation function
 */
class ReLU implements ActivationFunction {
    @Override
    public double apply(double x) {
        return Math.max(0, x);
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}

/**
 * Sigmoid activation function
 */
class Sigmoid implements ActivationFunction {
    @Override
    public double apply(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    @Override
    public double derivative(double x) {
        double sigmoid = apply(x);
        return sigmoid * (1.0 - sigmoid);
    }
}

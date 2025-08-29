# Chapter 7: Neural Networks

## Introduction

Neural networks are computational models inspired by biological neural networks in the human brain. They consist of interconnected nodes (neurons) organized in layers that can learn complex patterns from data. Neural networks form the foundation of deep learning and have revolutionized fields like computer vision, natural language processing, and speech recognition.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of neural networks and artificial neurons
- Implement feedforward neural networks from scratch
- Apply different activation functions and understand their properties
- Train neural networks using backpropagation and gradient descent
- Handle overfitting through regularization techniques
- Build and train neural networks for classification and regression tasks
- Choose appropriate network architectures for different problems

### Key Concepts

- **Artificial Neuron**: Basic computational unit that processes inputs and produces outputs
- **Activation Functions**: Non-linear functions that introduce complexity to neural networks
- **Feedforward Networks**: Networks where information flows from input to output layers
- **Backpropagation**: Algorithm for computing gradients and updating weights
- **Gradient Descent**: Optimization algorithm for minimizing loss functions
- **Regularization**: Techniques to prevent overfitting and improve generalization

## 7.1 Artificial Neurons

The artificial neuron (also called a perceptron) is the fundamental building block of neural networks. It takes multiple inputs, applies weights and a bias, and produces an output through an activation function.

### 7.1.1 Neuron Structure

A neuron computes: `output = activation_function(Σ(weight_i × input_i) + bias)`

#### Implementation

```java
package com.aiprogramming.ch07;

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
    
    // Private fields for storing gradients
    private List<Double> lastWeightGradients;
    private double lastBiasGradient;
}
```

### 7.1.2 Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns.

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.function.DoubleFunction;

/**
 * Activation function interface and implementations
 */
public interface ActivationFunction {
    double apply(double x);
    double derivative(double x);
}

/**
 * Sigmoid activation function: 1 / (1 + e^(-x))
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

/**
 * ReLU (Rectified Linear Unit) activation function: max(0, x)
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
 * Tanh (Hyperbolic Tangent) activation function: (e^x - e^(-x)) / (e^x + e^(-x))
 */
class Tanh implements ActivationFunction {
    @Override
    public double apply(double x) {
        return Math.tanh(x);
    }
    
    @Override
    public double derivative(double x) {
        double tanh = apply(x);
        return 1.0 - tanh * tanh;
    }
}

/**
 * Leaky ReLU activation function: max(0.01x, x)
 */
class LeakyReLU implements ActivationFunction {
    private final double alpha;
    
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }
    
    public LeakyReLU() {
        this(0.01);
    }
    
    @Override
    public double apply(double x) {
        return x > 0 ? x : alpha * x;
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : alpha;
    }
}

/**
 * Softmax activation function for output layers
 */
class Softmax implements ActivationFunction {
    private List<Double> lastOutputs;
    
    @Override
    public double apply(double x) {
        // Softmax is applied to a vector, not a single value
        // This is a placeholder - actual implementation is in the layer
        return x;
    }
    
    @Override
    public double derivative(double x) {
        // Derivative is computed differently for softmax
        return x;
    }
    
    public List<Double> applyToVector(List<Double> inputs) {
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
    
    public List<Double> derivativeForVector(List<Double> targets) {
        // Compute softmax derivative: softmax_i * (target_i - softmax_i)
        List<Double> derivatives = new ArrayList<>();
        for (int i = 0; i < lastOutputs.size(); i++) {
            derivatives.add(lastOutputs.get(i) * (targets.get(i) - lastOutputs.get(i)));
        }
        return derivatives;
    }
}
```

#### Key Features

- **Sigmoid**: Outputs values between 0 and 1, good for binary classification
- **ReLU**: Most popular activation function, helps with vanishing gradient problem
- **Tanh**: Outputs values between -1 and 1, zero-centered
- **Leaky ReLU**: Variant of ReLU that allows small negative gradients
- **Softmax**: Used in output layers for multi-class classification

## 7.2 Feedforward Neural Networks

Feedforward neural networks consist of layers of neurons where information flows from input to output without cycles.

### 7.2.1 Network Architecture

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.*;

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
    public List<List<Double>> predict(List<List<Double>> inputs) {
        List<List<Double>> predictions = new ArrayList<>();
        for (List<Double> input : inputs) {
            predictions.add(predict(input));
        }
        return predictions;
    }
}
```

### 7.2.2 Layer Implementation

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.*;

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
```

## 7.3 Loss Functions

Loss functions measure how well the network's predictions match the target values.

### 7.3.1 Common Loss Functions

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.List;

/**
 * Loss function interface
 */
public interface LossFunction {
    double compute(List<Double> predictions, List<Double> targets);
    List<Double> derivative(List<Double> predictions, List<Double> targets);
}

/**
 * Mean Squared Error loss function
 */
class MeanSquaredError implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double diff = predictions.get(i) - targets.get(i);
            sum += diff * diff;
        }
        
        return sum / predictions.size();
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double derivative = 2.0 * (predictions.get(i) - targets.get(i)) / predictions.size();
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}

/**
 * Cross-entropy loss function for classification
 */
class CrossEntropy implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            sum += -targets.get(i) * Math.log(pred) - (1.0 - targets.get(i)) * Math.log(1.0 - pred);
        }
        
        return sum / predictions.size();
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            double derivative = (pred - targets.get(i)) / (pred * (1.0 - pred)) / predictions.size();
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}

/**
 * Categorical cross-entropy for multi-class classification
 */
class CategoricalCrossEntropy implements LossFunction {
    @Override
    public double compute(List<Double> predictions, List<Double> targets) {
        if (predictions.size() != targets.size()) {
            throw new IllegalArgumentException("Predictions and targets must have same size");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            sum += -targets.get(i) * Math.log(pred);
        }
        
        return sum;
    }
    
    @Override
    public List<Double> derivative(List<Double> predictions, List<Double> targets) {
        List<Double> derivatives = new ArrayList<>();
        
        for (int i = 0; i < predictions.size(); i++) {
            double pred = Math.max(1e-15, Math.min(1.0 - 1e-15, predictions.get(i)));
            double derivative = -targets.get(i) / pred;
            derivatives.add(derivative);
        }
        
        return derivatives;
    }
}
```

## 7.4 Optimizers

Optimizers determine how the network weights are updated during training.

### 7.4.1 Gradient Descent Variants

#### Implementation

```java
package com.aiprogramming.ch07;

/**
 * Optimizer interface
 */
public interface Optimizer {
    double getLearningRate();
    void updateLearningRate(double newLearningRate);
}

/**
 * Stochastic Gradient Descent (SGD)
 */
class SGD implements Optimizer {
    private double learningRate;
    private double momentum;
    private List<Double> velocity;
    
    public SGD(double learningRate) {
        this(learningRate, 0.0);
    }
    
    public SGD(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocity = new ArrayList<>();
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
    
    @Override
    public void updateLearningRate(double newLearningRate) {
        this.learningRate = newLearningRate;
    }
    
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
}

/**
 * Adam optimizer
 */
class Adam implements Optimizer {
    private double learningRate;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int t;
    private List<Double> m; // First moment
    private List<Double> v; // Second moment
    
    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }
    
    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.m = new ArrayList<>();
        this.v = new ArrayList<>();
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
    
    @Override
    public void updateLearningRate(double newLearningRate) {
        this.learningRate = newLearningRate;
    }
}
```

## 7.5 Training Neural Networks

### 7.5.1 Backpropagation Algorithm

Backpropagation is the algorithm used to compute gradients for updating network weights.

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.*;

/**
 * Training utilities for neural networks
 */
public class NeuralNetworkTrainer {
    
    /**
     * Train a neural network with mini-batch gradient descent
     */
    public static List<Double> trainWithMiniBatches(NeuralNetwork network, 
                                                   List<List<Double>> inputs, 
                                                   List<List<Double>> targets, 
                                                   int epochs, 
                                                   int batchSize) {
        List<Double> losses = new ArrayList<>();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;
            int batchCount = 0;
            
            // Create mini-batches
            for (int i = 0; i < inputs.size(); i += batchSize) {
                int endIndex = Math.min(i + batchSize, inputs.size());
                List<List<Double>> batchInputs = inputs.subList(i, endIndex);
                List<List<Double>> batchTargets = targets.subList(i, endIndex);
                
                // Train on batch
                double batchLoss = network.trainEpoch(batchInputs, batchTargets);
                epochLoss += batchLoss;
                batchCount++;
            }
            
            double averageLoss = epochLoss / batchCount;
            losses.add(averageLoss);
            
            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d, Average Loss: %.6f%n", epoch, averageLoss);
            }
        }
        
        return losses;
    }
    
    /**
     * Split data into training and validation sets
     */
    public static DataSplit splitData(List<List<Double>> inputs, 
                                     List<List<Double>> targets, 
                                     double trainRatio) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Input and target sizes must match");
        }
        
        int totalSize = inputs.size();
        int trainSize = (int) (totalSize * trainRatio);
        
        // Shuffle data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSize; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        
        // Split data
        List<List<Double>> trainInputs = new ArrayList<>();
        List<List<Double>> trainTargets = new ArrayList<>();
        List<List<Double>> valInputs = new ArrayList<>();
        List<List<Double>> valTargets = new ArrayList<>();
        
        for (int i = 0; i < trainSize; i++) {
            int index = indices.get(i);
            trainInputs.add(inputs.get(index));
            trainTargets.add(targets.get(index));
        }
        
        for (int i = trainSize; i < totalSize; i++) {
            int index = indices.get(i);
            valInputs.add(inputs.get(index));
            valTargets.add(targets.get(index));
        }
        
        return new DataSplit(trainInputs, trainTargets, valInputs, valTargets);
    }
    
    /**
     * Early stopping to prevent overfitting
     */
    public static NeuralNetwork trainWithEarlyStopping(NeuralNetwork network,
                                                      List<List<Double>> trainInputs,
                                                      List<List<Double>> trainTargets,
                                                      List<List<Double>> valInputs,
                                                      List<List<Double>> valTargets,
                                                      int maxEpochs,
                                                      int patience) {
        double bestValLoss = Double.MAX_VALUE;
        int epochsWithoutImprovement = 0;
        NeuralNetwork bestNetwork = null; // In practice, you'd save the best weights
        
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // Train for one epoch
            double trainLoss = network.trainEpoch(trainInputs, trainTargets);
            
            // Evaluate on validation set
            double valLoss = evaluateLoss(network, valInputs, valTargets);
            
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                epochsWithoutImprovement = 0;
                bestNetwork = network; // In practice, save weights here
            } else {
                epochsWithoutImprovement++;
            }
            
            if (epochsWithoutImprovement >= patience) {
                System.out.printf("Early stopping at epoch %d%n", epoch);
                break;
            }
            
            if (epoch % 100 == 0) {
                System.out.printf("Epoch %d, Train Loss: %.6f, Val Loss: %.6f%n", 
                                epoch, trainLoss, valLoss);
            }
        }
        
        return bestNetwork != null ? bestNetwork : network;
    }
    
    /**
     * Evaluate loss on a dataset
     */
    private static double evaluateLoss(NeuralNetwork network, 
                                     List<List<Double>> inputs, 
                                     List<List<Double>> targets) {
        double totalLoss = 0.0;
        
        for (int i = 0; i < inputs.size(); i++) {
            List<Double> predictions = network.predict(inputs.get(i));
            totalLoss += network.getLossFunction().compute(predictions, targets.get(i));
        }
        
        return totalLoss / inputs.size();
    }
    
    /**
     * Data split container
     */
    public static class DataSplit {
        public final List<List<Double>> trainInputs;
        public final List<List<Double>> trainTargets;
        public final List<List<Double>> valInputs;
        public final List<List<Double>> valTargets;
        
        public DataSplit(List<List<Double>> trainInputs, List<List<Double>> trainTargets,
                        List<List<Double>> valInputs, List<List<Double>> valTargets) {
            this.trainInputs = trainInputs;
            this.trainTargets = trainTargets;
            this.valInputs = valInputs;
            this.valTargets = valTargets;
        }
    }
}
```

## 7.6 Regularization Techniques

Regularization helps prevent overfitting by adding constraints to the model.

### 7.6.1 L1 and L2 Regularization

#### Implementation

```java
package com.aiprogramming.ch07;

/**
 * Regularization techniques for neural networks
 */
public class Regularization {
    
    /**
     * L2 regularization (weight decay)
     */
    public static double computeL2Regularization(List<Layer> layers, double lambda) {
        double regularization = 0.0;
        
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                for (Neuron neuron : denseLayer.getNeurons()) {
                    for (Double weight : neuron.getWeights()) {
                        regularization += weight * weight;
                    }
                }
            }
        }
        
        return 0.5 * lambda * regularization;
    }
    
    /**
     * L1 regularization
     */
    public static double computeL1Regularization(List<Layer> layers, double lambda) {
        double regularization = 0.0;
        
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                for (Neuron neuron : denseLayer.getNeurons()) {
                    for (Double weight : neuron.getWeights()) {
                        regularization += Math.abs(weight);
                    }
                }
            }
        }
        
        return lambda * regularization;
    }
    
    /**
     * Add regularization gradients to weight gradients
     */
    public static void addRegularizationGradients(List<Layer> layers, 
                                                 List<List<Double>> weightGradients, 
                                                 double lambda, 
                                                 String type) {
        int gradientIndex = 0;
        
        for (Layer layer : layers) {
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                for (Neuron neuron : denseLayer.getNeurons()) {
                    List<Double> neuronWeights = neuron.getWeights();
                    List<Double> neuronGradients = weightGradients.get(gradientIndex);
                    
                    for (int i = 0; i < neuronWeights.size(); i++) {
                        if (type.equals("L2")) {
                            neuronGradients.set(i, neuronGradients.get(i) + lambda * neuronWeights.get(i));
                        } else if (type.equals("L1")) {
                            double sign = neuronWeights.get(i) > 0 ? 1.0 : (neuronWeights.get(i) < 0 ? -1.0 : 0.0);
                            neuronGradients.set(i, neuronGradients.get(i) + lambda * sign);
                        }
                    }
                    
                    gradientIndex++;
                }
            }
        }
    }
}
```

## 7.7 Practical Examples

### 7.7.1 XOR Problem

The XOR problem is a classic example that demonstrates the need for hidden layers in neural networks.

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.*;

/**
 * XOR problem demonstration
 */
public class XORExample {
    
    public static void main(String[] args) {
        // Create XOR dataset
        List<List<Double>> inputs = Arrays.asList(
            Arrays.asList(0.0, 0.0),
            Arrays.asList(0.0, 1.0),
            Arrays.asList(1.0, 0.0),
            Arrays.asList(1.0, 1.0)
        );
        
        List<List<Double>> targets = Arrays.asList(
            Arrays.asList(0.0),
            Arrays.asList(1.0),
            Arrays.asList(1.0),
            Arrays.asList(0.0)
        );
        
        // Create neural network
        NeuralNetwork network = new NeuralNetwork();
        
        // Add layers
        network.addLayer(new DenseLayer(2, 4, new ReLU())); // Hidden layer
        network.addLayer(new DenseLayer(4, 1, new Sigmoid())); // Output layer
        
        // Set loss function
        network.setLossFunction(new MeanSquaredError());
        
        // Train the network
        List<Double> losses = network.train(inputs, targets, 10000);
        
        // Test the network
        System.out.println("XOR Results:");
        for (int i = 0; i < inputs.size(); i++) {
            List<Double> prediction = network.predict(inputs.get(i));
            System.out.printf("Input: %s, Target: %.1f, Prediction: %.3f%n", 
                            inputs.get(i), targets.get(i).get(0), prediction.get(0));
        }
    }
}
```

### 7.7.2 Digit Classification

A more complex example using neural networks for digit classification.

#### Implementation

```java
package com.aiprogramming.ch07;

import java.util.*;

/**
 * Digit classification example
 */
public class DigitClassificationExample {
    
    public static void main(String[] args) {
        // Load digit dataset (simplified)
        List<List<Double>> inputs = loadDigitData();
        List<List<Double>> targets = createOneHotTargets();
        
        // Split data
        DataSplit split = NeuralNetworkTrainer.splitData(inputs, targets, 0.8);
        
        // Create neural network
        NeuralNetwork network = new NeuralNetwork();
        
        // Add layers with dropout for regularization
        network.addLayer(new DenseLayer(784, 128, new ReLU())); // Input layer (28x28 = 784)
        network.addLayer(new DropoutLayer(0.3));
        network.addLayer(new DenseLayer(128, 64, new ReLU())); // Hidden layer
        network.addLayer(new DropoutLayer(0.3));
        network.addLayer(new DenseLayer(64, 10, new ReLU())); // Output layer (10 digits)
        network.addLayer(new SoftmaxLayer()); // Softmax activation
        
        // Set loss function for multi-class classification
        network.setLossFunction(new CategoricalCrossEntropy());
        
        // Train with early stopping
        NeuralNetwork bestNetwork = NeuralNetworkTrainer.trainWithEarlyStopping(
            network, split.trainInputs, split.trainTargets, 
            split.valInputs, split.valTargets, 1000, 50);
        
        // Evaluate on test set
        double accuracy = evaluateAccuracy(bestNetwork, split.valInputs, split.valTargets);
        System.out.printf("Validation Accuracy: %.2f%%%n", accuracy * 100);
    }
    
    private static List<List<Double>> loadDigitData() {
        // In practice, load from MNIST dataset
        // For this example, return dummy data
        List<List<Double>> data = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < 1000; i++) {
            List<Double> sample = new ArrayList<>();
            for (int j = 0; j < 784; j++) {
                sample.add(random.nextDouble());
            }
            data.add(sample);
        }
        
        return data;
    }
    
    private static List<List<Double>> createOneHotTargets() {
        // Create one-hot encoded targets
        List<List<Double>> targets = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < 1000; i++) {
            List<Double> target = new ArrayList<>();
            int digit = random.nextInt(10);
            
            for (int j = 0; j < 10; j++) {
                target.add(j == digit ? 1.0 : 0.0);
            }
            targets.add(target);
        }
        
        return targets;
    }
    
    private static double evaluateAccuracy(NeuralNetwork network, 
                                         List<List<Double>> inputs, 
                                         List<List<Double>> targets) {
        int correct = 0;
        
        for (int i = 0; i < inputs.size(); i++) {
            List<Double> prediction = network.predict(inputs.get(i));
            List<Double> target = targets.get(i);
            
            // Find predicted class
            int predictedClass = 0;
            double maxProb = prediction.get(0);
            for (int j = 1; j < prediction.size(); j++) {
                if (prediction.get(j) > maxProb) {
                    maxProb = prediction.get(j);
                    predictedClass = j;
                }
            }
            
            // Find true class
            int trueClass = 0;
            for (int j = 0; j < target.size(); j++) {
                if (target.get(j) == 1.0) {
                    trueClass = j;
                    break;
                }
            }
            
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        
        return (double) correct / inputs.size();
    }
}
```

## 7.8 Best Practices

### 7.8.1 Network Architecture

- **Start Simple**: Begin with a simple architecture and gradually increase complexity
- **Layer Sizes**: Use decreasing layer sizes (e.g., 784 → 128 → 64 → 10)
- **Activation Functions**: Use ReLU for hidden layers, appropriate activation for output layer
- **Regularization**: Add dropout layers to prevent overfitting

### 7.8.2 Training

- **Learning Rate**: Start with a small learning rate (0.01) and adjust based on convergence
- **Batch Size**: Use mini-batches for better generalization and faster training
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Data Preprocessing**: Normalize inputs to similar ranges

### 7.8.3 Evaluation

- **Cross-Validation**: Use k-fold cross-validation for robust evaluation
- **Multiple Metrics**: Use accuracy, precision, recall, and F1-score for classification
- **Confusion Matrix**: Analyze misclassifications to understand model behavior

## 7.9 Summary

In this chapter, we explored the fundamentals of neural networks:

1. **Artificial Neurons**: Basic computational units with weights, bias, and activation functions
2. **Feedforward Networks**: Multi-layer networks for pattern recognition
3. **Activation Functions**: Non-linear functions that enable complex mappings
4. **Training Algorithms**: Backpropagation and gradient descent for learning
5. **Regularization**: Techniques to prevent overfitting and improve generalization

### Key Takeaways

- **Neural networks** are powerful function approximators
- **Backpropagation** efficiently computes gradients for weight updates
- **Activation functions** introduce non-linearity and enable complex mappings
- **Regularization** is crucial for preventing overfitting
- **Architecture design** requires careful consideration of problem requirements

### Next Steps

- Explore convolutional neural networks for image processing
- Learn about recurrent neural networks for sequential data
- Implement advanced optimization techniques
- Apply neural networks to real-world problems

## Exercises

### Exercise 1: Basic Neural Network
Implement a neural network with one hidden layer to solve the XOR problem. Experiment with different activation functions and network architectures.

### Exercise 2: Multi-class Classification
Build a neural network for classifying handwritten digits using the MNIST dataset. Implement dropout regularization and early stopping.

### Exercise 3: Regression Problem
Create a neural network for predicting house prices. Use appropriate loss functions and evaluate performance using mean squared error.

### Exercise 4: Activation Function Comparison
Compare the performance of different activation functions (Sigmoid, ReLU, Tanh, Leaky ReLU) on a classification task. Analyze convergence speed and final accuracy.

### Exercise 5: Regularization Study
Implement L1 and L2 regularization in your neural network. Compare their effects on model performance and weight distributions.

### Exercise 6: Hyperparameter Tuning
Perform hyperparameter optimization for learning rate, batch size, and network architecture. Use cross-validation to find optimal parameters.

### Exercise 7: Visualizing Training
Create plots showing training and validation loss over time. Implement learning rate scheduling and visualize its effects.

### Exercise 8: Transfer Learning
Implement a pre-trained neural network and fine-tune it for a new task. Compare performance with training from scratch.

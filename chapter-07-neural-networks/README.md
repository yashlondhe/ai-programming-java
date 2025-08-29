# Chapter 7: Neural Networks

This chapter implements neural networks from scratch in Java, covering the fundamental concepts of artificial neural networks, backpropagation, and deep learning.

## Overview

This project provides a complete implementation of feedforward neural networks with the following features:

- **Artificial Neurons**: Basic computational units with weights, bias, and activation functions
- **Activation Functions**: Sigmoid, ReLU, Tanh, Leaky ReLU, and Softmax
- **Network Layers**: Dense (fully connected) layers and Dropout layers for regularization
- **Loss Functions**: Mean Squared Error, Cross-entropy, and Categorical Cross-entropy
- **Optimizers**: Stochastic Gradient Descent (SGD) and Adam
- **Training Utilities**: Mini-batch training, early stopping, and data splitting
- **Regularization**: L1 and L2 regularization techniques

## Project Structure

```
src/main/java/com/aiprogramming/ch07/
├── Neuron.java                    # Artificial neuron implementation
├── ActivationFunction.java        # Activation function interface and implementations
├── Layer.java                     # Abstract layer and concrete implementations
├── LossFunction.java              # Loss function interface and implementations
├── Optimizer.java                 # Optimizer interface and implementations
├── NeuralNetwork.java             # Main neural network class
├── NeuralNetworkTrainer.java      # Training utilities
├── Regularization.java            # Regularization techniques
├── XORExample.java                # XOR problem demonstration
├── SimpleClassificationExample.java # Simple classification example
└── DigitClassificationExample.java # Digit classification example
```

## Key Components

### 1. Neuron
The basic computational unit that processes inputs and produces outputs through an activation function.

### 2. Activation Functions
- **Sigmoid**: Outputs values between 0 and 1
- **ReLU**: Most popular activation function, helps with vanishing gradient problem
- **Tanh**: Outputs values between -1 and 1, zero-centered
- **Leaky ReLU**: Variant of ReLU that allows small negative gradients
- **Softmax**: Used in output layers for multi-class classification

### 3. Layers
- **DenseLayer**: Fully connected layer with neurons
- **DropoutLayer**: Regularization layer that randomly drops neurons during training

### 4. Loss Functions
- **MeanSquaredError**: For regression problems
- **CrossEntropy**: For binary classification
- **CategoricalCrossEntropy**: For multi-class classification

### 5. Optimizers
- **SGD**: Stochastic Gradient Descent with optional momentum
- **Adam**: Adaptive moment estimation optimizer

## Usage Examples

### XOR Problem
```java
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

// Create and train network
NeuralNetwork network = new NeuralNetwork();
network.addLayer(new DenseLayer(2, 4, new ReLU()));
network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
network.setLossFunction(new MeanSquaredError());

List<Double> losses = network.train(inputs, targets, 10000);
```

### Digit Classification
```java
// Load and split data
NeuralNetworkTrainer.DataSplit split = NeuralNetworkTrainer.splitData(inputs, targets, 0.8);

// Create network with dropout
NeuralNetwork network = new NeuralNetwork();
network.addLayer(new DenseLayer(784, 128, new ReLU()));
network.addLayer(new DropoutLayer(0.3));
network.addLayer(new DenseLayer(128, 64, new ReLU()));
network.addLayer(new DropoutLayer(0.3));
network.addLayer(new DenseLayer(64, 10, new Softmax()));

// Train with early stopping
NeuralNetwork bestNetwork = NeuralNetworkTrainer.trainWithEarlyStopping(
    network, split.trainInputs, split.trainTargets, 
    split.valInputs, split.valTargets, 1000, 50);
```

## Building and Running

### Prerequisites
- Java 11 or higher
- Maven 3.6 or higher

### Build the Project
```bash
mvn clean compile
```

### Run Examples
```bash
# Run XOR example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch07.XORExample"

# Run simple classification example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch07.SimpleClassificationExample"

# Run digit classification example (may require additional setup)
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch07.DigitClassificationExample"
```

### Run Tests
```bash
mvn test
```

## Key Concepts Covered

1. **Forward Propagation**: How information flows through the network
2. **Backpropagation**: Algorithm for computing gradients and updating weights
3. **Gradient Descent**: Optimization algorithm for minimizing loss functions
4. **Regularization**: Techniques to prevent overfitting (Dropout, L1/L2)
5. **Activation Functions**: Non-linear functions that enable complex mappings
6. **Loss Functions**: How to measure prediction accuracy
7. **Optimizers**: Different strategies for updating network weights

## Learning Objectives

By working with this code, you will understand:

- How artificial neurons compute outputs
- Why activation functions are necessary
- How backpropagation works
- The role of different loss functions
- How regularization prevents overfitting
- The importance of proper network architecture
- How to train neural networks effectively

## Exercises

The chapter includes several exercises to reinforce learning:

1. **Basic Neural Network**: Implement XOR with different architectures
2. **Multi-class Classification**: Build a digit classifier
3. **Regression Problem**: Predict house prices
4. **Activation Function Comparison**: Compare different activation functions
5. **Regularization Study**: Implement and compare L1/L2 regularization
6. **Hyperparameter Tuning**: Optimize learning rate and architecture
7. **Training Visualization**: Plot training curves
8. **Transfer Learning**: Fine-tune pre-trained networks

## Next Steps

After mastering this chapter, you can explore:

- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) for sequential data
- Advanced optimization techniques
- Deep learning frameworks like TensorFlow or PyTorch
- Real-world applications of neural networks

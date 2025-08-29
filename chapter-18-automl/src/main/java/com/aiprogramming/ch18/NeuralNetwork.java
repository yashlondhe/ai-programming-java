package com.aiprogramming.ch18;

import java.util.*;

/**
 * Simple Neural Network model
 */
public class NeuralNetwork implements MLModel {
    
    private List<Layer> layers;
    private Map<String, Object> hyperparameters;
    
    public NeuralNetwork() {
        this.hyperparameters = new HashMap<>();
        this.hyperparameters.put("learningRate", 0.01);
        this.hyperparameters.put("hiddenLayers", 1);
        this.hyperparameters.put("neuronsPerLayer", 10);
        this.hyperparameters.put("dropout", 0.0);
        this.layers = new ArrayList<>();
    }
    
    @Override
    public void train(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        int hiddenLayers = (Integer) hyperparameters.get("hiddenLayers");
        int neuronsPerLayer = (Integer) hyperparameters.get("neuronsPerLayer");
        double learningRate = (Double) hyperparameters.get("learningRate");
        
        // Build network architecture
        layers.clear();
        
        // Input layer
        if (hiddenLayers > 0) {
            layers.add(new Layer(numFeatures, neuronsPerLayer));
            for (int i = 1; i < hiddenLayers; i++) {
                layers.add(new Layer(neuronsPerLayer, neuronsPerLayer));
            }
            layers.add(new Layer(neuronsPerLayer, 1)); // Output layer
        } else {
            layers.add(new Layer(numFeatures, 1)); // Direct input to output
        }
        
        // Train using backpropagation
        int maxIterations = 1000;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < features.length; i++) {
                // Forward pass
                double[] activations = forwardPass(features[i]);
                
                // Calculate loss
                double error = activations[activations.length - 1] - targets[i];
                totalLoss += error * error;
                
                // Backward pass
                backpropagate(features[i], targets[i], learningRate);
            }
            
            // Check for NaN or infinite values
            if (Double.isNaN(totalLoss) || Double.isInfinite(totalLoss)) {
                System.out.printf("Iteration %d: Loss became NaN/Infinite, stopping training%n", iteration);
                break;
            }
            
            if (iteration % 100 == 0) {
                System.out.printf("Iteration %d, Loss: %.4f%n", iteration, totalLoss / features.length);
            }
        }
    }
    
    @Override
    public double[] predict(double[][] features) {
        double[] predictions = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            double[] activations = forwardPass(features[i]);
            predictions[i] = activations[activations.length - 1];
        }
        return predictions;
    }
    
    @Override
    public double evaluate(double[][] features, double[] targets) {
        double[] predictions = predict(features);
        double mse = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - targets[i];
            mse += error * error;
        }
        return mse / predictions.length;
    }
    
    @Override
    public void setHyperparameters(Map<String, Object> hyperparameters) {
        this.hyperparameters.putAll(hyperparameters);
    }
    
    @Override
    public Map<String, Object> getHyperparameters() {
        return new HashMap<>(hyperparameters);
    }
    
    private double[] forwardPass(double[] input) {
        double[] currentActivations = input;
        
        for (Layer layer : layers) {
            currentActivations = layer.forward(currentActivations);
        }
        
        return currentActivations;
    }
    
    private void backpropagate(double[] input, double target, double learningRate) {
        // Forward pass to get all activations
        List<double[]> activations = new ArrayList<>();
        double[] currentActivations = input;
        activations.add(currentActivations);
        
        for (Layer layer : layers) {
            currentActivations = layer.forward(currentActivations);
            activations.add(currentActivations);
        }
        
        // Backward pass
        double[] delta = new double[1];
        delta[0] = activations.get(activations.size() - 1)[0] - target;
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            double[] prevActivations = activations.get(i);
            
            // Update weights
            layer.updateWeights(prevActivations, delta, learningRate);
            
            // Calculate delta for next layer
            if (i > 0) {
                delta = layer.backpropagate(delta);
            }
        }
    }
    
    /**
     * Simple neural network layer
     */
    private static class Layer {
        private double[][] weights;
        private double[] biases;
        private final int inputSize;
        private final int outputSize;
        
        public Layer(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = new double[inputSize][outputSize];
            this.biases = new double[outputSize];
            
            // Initialize weights randomly
            Random random = new Random();
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[i][j] = random.nextGaussian() * 0.1;
                }
            }
            
            for (int j = 0; j < outputSize; j++) {
                biases[j] = random.nextGaussian() * 0.1;
            }
        }
        
        public double[] forward(double[] input) {
            double[] output = new double[outputSize];
            
            for (int j = 0; j < outputSize; j++) {
                double sum = biases[j];
                for (int i = 0; i < inputSize; i++) {
                    sum += weights[i][j] * input[i];
                }
                // Use tanh instead of relu to avoid exploding gradients
                output[j] = Math.tanh(sum);
            }
            
            return output;
        }
        
        public void updateWeights(double[] input, double[] delta, double learningRate) {
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[i][j] -= learningRate * delta[j] * input[i];
                }
            }
            
            for (int j = 0; j < outputSize; j++) {
                biases[j] -= learningRate * delta[j];
            }
        }
        
        public double[] backpropagate(double[] delta) {
            double[] prevDelta = new double[inputSize];
            
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    prevDelta[i] += weights[i][j] * delta[j];
                }
            }
            
            return prevDelta;
        }
        
        private double relu(double x) {
            return Math.max(0, x);
        }
        
        private double tanhDerivative(double x) {
            return 1.0 - x * x;
        }
    }
}

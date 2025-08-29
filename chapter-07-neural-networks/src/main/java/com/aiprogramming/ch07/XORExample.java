package com.aiprogramming.ch07;

import com.aiprogramming.utils.*;
import java.util.Arrays;
import java.util.List;

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
        
        // Using MatrixUtils for weight initialization demonstration
        System.out.println("Matrix Operations with Utils:");
        double[][] weights1 = MatrixUtils.randomNormal(2, 4, 0.0, 0.1, 42L);
        double[][] weights2 = MatrixUtils.randomNormal(4, 1, 0.0, 0.1, 43L);
        
        System.out.println("Weight matrices initialized with utils:");
        DataUtils.printMatrix(weights1, "Hidden Layer Weights");
        DataUtils.printMatrix(weights2, "Output Layer Weights");
        
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

package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Simple classification example using neural networks
 */
public class SimpleClassificationExample {
    
    public static void main(String[] args) {
        // Create a simple 2D classification dataset
        List<List<Double>> inputs = generateClassificationData();
        List<List<Double>> targets = generateClassificationTargets();
        
        // Split data
        NeuralNetworkTrainer.DataSplit split = NeuralNetworkTrainer.splitData(inputs, targets, 0.8);
        
        // Create neural network
        NeuralNetwork network = new NeuralNetwork();
        
        // Add layers
        network.addLayer(new DenseLayer(2, 8, new ReLU())); // Input layer
        network.addLayer(new DenseLayer(8, 4, new ReLU())); // Hidden layer
        network.addLayer(new DenseLayer(4, 1, new Sigmoid())); // Output layer
        
        // Set loss function for binary classification
        network.setLossFunction(new CrossEntropy());
        
        // Train the network
        System.out.println("Training neural network for classification...");
        List<Double> losses = network.train(split.trainInputs, split.trainTargets, 1000);
        
        // Evaluate on validation set
        double accuracy = evaluateAccuracy(network, split.valInputs, split.valTargets);
        System.out.printf("Validation Accuracy: %.2f%%%n", accuracy * 100);
        
        // Test some predictions
        System.out.println("\nSample Predictions:");
        for (int i = 0; i < Math.min(10, split.valInputs.size()); i++) {
            List<Double> prediction = network.predict(split.valInputs.get(i));
            List<Double> target = split.valTargets.get(i);
            System.out.printf("Input: [%.2f, %.2f], Target: %.1f, Prediction: %.3f%n", 
                            split.valInputs.get(i).get(0), split.valInputs.get(i).get(1),
                            target.get(0), prediction.get(0));
        }
    }
    
    private static List<List<Double>> generateClassificationData() {
        List<List<Double>> data = new ArrayList<>();
        Random random = new Random();
        
        // Generate 200 data points
        for (int i = 0; i < 200; i++) {
            List<Double> point = new ArrayList<>();
            // Generate points in a circle pattern
            double angle = random.nextDouble() * 2 * Math.PI;
            double radius = random.nextDouble() * 2 + 1; // Radius between 1 and 3
            
            point.add(Math.cos(angle) * radius);
            point.add(Math.sin(angle) * radius);
            data.add(point);
        }
        
        return data;
    }
    
    private static List<List<Double>> generateClassificationTargets() {
        List<List<Double>> targets = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < 200; i++) {
            List<Double> target = new ArrayList<>();
            // Simple classification: points with radius > 2 are class 1, others are class 0
            double radius = Math.sqrt(Math.pow(random.nextDouble() * 2 + 1, 2) + 
                                    Math.pow(random.nextDouble() * 2 + 1, 2));
            target.add(radius > 2 ? 1.0 : 0.0);
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
            
            // Binary classification: threshold at 0.5
            int predictedClass = prediction.get(0) > 0.5 ? 1 : 0;
            int trueClass = target.get(0) > 0.5 ? 1 : 0;
            
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        
        return (double) correct / inputs.size();
    }
}

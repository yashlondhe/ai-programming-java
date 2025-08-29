package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Digit classification example
 */
public class DigitClassificationExample {
    
    public static void main(String[] args) {
        // Load digit dataset (simplified)
        List<List<Double>> inputs = loadDigitData();
        List<List<Double>> targets = createOneHotTargets();
        
        // Split data
        NeuralNetworkTrainer.DataSplit split = NeuralNetworkTrainer.splitData(inputs, targets, 0.8);
        
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

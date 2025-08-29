package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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

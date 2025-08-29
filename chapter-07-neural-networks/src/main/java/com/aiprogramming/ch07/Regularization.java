package com.aiprogramming.ch07;

import java.util.List;

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

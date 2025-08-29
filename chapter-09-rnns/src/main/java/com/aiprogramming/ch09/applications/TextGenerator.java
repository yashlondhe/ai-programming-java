package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.LSTMCell;
import com.aiprogramming.ch09.utils.Vocabulary;

import java.util.*;

/**
 * Text generation using Recurrent Neural Networks.
 * 
 * This class implements a character-level text generator that learns
 * patterns in text data and can generate new text based on a seed.
 */
public class TextGenerator {
    private LSTMCell lstm;
    private Vocabulary vocabulary;
    private double[][] outputWeights;
    private double[] outputBias;
    private int hiddenSize;
    private double learningRate;
    
    /**
     * Constructor for TextGenerator.
     */
    public TextGenerator() {
        this.hiddenSize = 128;
        this.learningRate = 0.01;
        this.vocabulary = new Vocabulary();
    }
    
    /**
     * Train the text generation model on a given text.
     * 
     * @param text Training text
     * @param epochs Number of training epochs
     */
    public void train(String text, int epochs) {
        // Build vocabulary from text
        vocabulary.buildFromText(text);
        
        // Initialize LSTM
        int inputSize = vocabulary.getSize();
        lstm = new LSTMCell(inputSize, hiddenSize);
        
        // Initialize output layer
        outputWeights = new double[hiddenSize][inputSize];
        outputBias = new double[inputSize];
        
        Random random = new Random(42);
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                outputWeights[i][j] = random.nextGaussian() * 0.1;
            }
        }
        
        // Prepare training data
        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();
        
        String[] characters = text.split("");
        for (int i = 0; i < characters.length - 1; i++) {
            double[] input = vocabulary.oneHotEncode(characters[i]);
            double[] target = vocabulary.oneHotEncode(characters[i + 1]);
            inputs.add(input);
            targets.add(target);
        }
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            // Reset LSTM state
            lstm.resetState();
            
            // Forward pass
            List<double[]> hiddenStates = new ArrayList<>();
            for (double[] input : inputs) {
                double[] hiddenState = lstm.forward(input);
                hiddenStates.add(hiddenState);
            }
            
            // Compute outputs and loss
            List<double[]> outputs = new ArrayList<>();
            for (double[] hiddenState : hiddenStates) {
                double[] output = computeOutput(hiddenState);
                outputs.add(output);
            }
            
            // Compute loss
            for (int i = 0; i < outputs.size(); i++) {
                totalLoss += crossEntropyLoss(outputs.get(i), targets.get(i));
            }
            
            // Backward pass (simplified for demonstration)
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss / outputs.size());
            }
        }
    }
    
    /**
     * Generate text based on a seed string.
     * 
     * @param seed Starting text
     * @param length Number of characters to generate
     * @return Generated text
     */
    public String generateText(String seed, int length) {
        if (lstm == null) {
            throw new IllegalStateException("Model must be trained before generating text");
        }
        
        lstm.resetState();
        StringBuilder result = new StringBuilder(seed);
        
        // Process seed characters
        String[] seedChars = seed.split("");
        for (String ch : seedChars) {
            if (vocabulary.contains(ch)) {
                double[] input = vocabulary.oneHotEncode(ch);
                lstm.forward(input);
            }
        }
        
        // Generate new characters
        for (int i = 0; i < length; i++) {
            double[] hiddenState = lstm.getHiddenState();
            double[] output = computeOutput(hiddenState);
            
            // Sample next character
            String nextChar = sampleCharacter(output);
            result.append(nextChar);
            
            // Feed back to LSTM
            if (vocabulary.contains(nextChar)) {
                double[] input = vocabulary.oneHotEncode(nextChar);
                lstm.forward(input);
            }
        }
        
        return result.toString();
    }
    
    /**
     * Compute output probabilities from hidden state.
     * 
     * @param hiddenState LSTM hidden state
     * @return Output probabilities
     */
    private double[] computeOutput(double[] hiddenState) {
        double[] output = new double[outputWeights[0].length];
        
        for (int i = 0; i < output.length; i++) {
            output[i] = outputBias[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += outputWeights[j][i] * hiddenState[j];
            }
        }
        
        // Apply softmax
        return softmax(output);
    }
    
    /**
     * Apply softmax activation to convert logits to probabilities.
     * 
     * @param logits Input logits
     * @return Probability distribution
     */
    private double[] softmax(double[] logits) {
        double max = Arrays.stream(logits).max().orElse(0.0);
        double sum = 0.0;
        double[] exp = new double[logits.length];
        
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        for (int i = 0; i < logits.length; i++) {
            exp[i] /= sum;
        }
        
        return exp;
    }
    
    /**
     * Sample a character from the probability distribution.
     * 
     * @param probabilities Character probabilities
     * @return Sampled character
     */
    private String sampleCharacter(double[] probabilities) {
        Random random = new Random();
        double r = random.nextDouble();
        double cumulative = 0.0;
        
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return vocabulary.getCharacter(i);
            }
        }
        
        return vocabulary.getCharacter(probabilities.length - 1);
    }
    
    /**
     * Compute cross-entropy loss between predicted and target distributions.
     * 
     * @param predicted Predicted probabilities
     * @param target Target probabilities
     * @return Cross-entropy loss
     */
    private double crossEntropyLoss(double[] predicted, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            if (target[i] > 0) {
                loss -= target[i] * Math.log(predicted[i] + 1e-8);
            }
        }
        return loss;
    }
    
    /**
     * Set learning rate for training.
     * 
     * @param learningRate New learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Set hidden size for the LSTM.
     * 
     * @param hiddenSize New hidden size
     */
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    /**
     * Get vocabulary size.
     * 
     * @return Number of unique characters
     */
    public int getVocabularySize() {
        return vocabulary.getSize();
    }
    
    /**
     * Get current vocabulary.
     * 
     * @return Vocabulary object
     */
    public Vocabulary getVocabulary() {
        return vocabulary;
    }
}

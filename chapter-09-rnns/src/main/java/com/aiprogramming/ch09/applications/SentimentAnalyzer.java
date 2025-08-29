package com.aiprogramming.ch09.applications;

import com.aiprogramming.ch09.core.GRUCell;
import com.aiprogramming.ch09.utils.Vocabulary;

import java.util.*;

/**
 * Sentiment analysis using Recurrent Neural Networks.
 * 
 * This class implements a sentiment analyzer that can classify text
 * as positive, negative, or neutral based on learned patterns.
 */
public class SentimentAnalyzer {
    private GRUCell gru;
    private Vocabulary vocabulary;
    private double[][] outputWeights;
    private double[] outputBias;
    private int hiddenSize;
    private double learningRate;
    private boolean isTrained;
    
    // Simple sentiment lexicon for demonstration
    private Map<String, Double> positiveWords;
    private Map<String, Double> negativeWords;
    
    /**
     * Constructor for SentimentAnalyzer.
     */
    public SentimentAnalyzer() {
        this.hiddenSize = 64;
        this.learningRate = 0.01;
        this.vocabulary = new Vocabulary();
        this.isTrained = false;
        
        initializeSentimentLexicon();
    }
    
    /**
     * Initialize a simple sentiment lexicon for demonstration.
     */
    private void initializeSentimentLexicon() {
        positiveWords = new HashMap<>();
        negativeWords = new HashMap<>();
        
        // Positive words
        String[] positive = {
            "love", "great", "good", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "brilliant", "outstanding", "perfect", "best", "favorite",
            "enjoy", "like", "happy", "pleased", "satisfied", "recommend", "highly"
        };
        
        // Negative words
        String[] negative = {
            "hate", "terrible", "bad", "awful", "horrible", "disappointing", "worst",
            "terrible", "disgusting", "annoying", "frustrated", "angry", "upset",
            "dislike", "poor", "unhappy", "dissatisfied", "avoid", "never", "worst"
        };
        
        for (String word : positive) {
            positiveWords.put(word.toLowerCase(), 1.0);
        }
        
        for (String word : negative) {
            negativeWords.put(word.toLowerCase(), -1.0);
        }
    }
    
    /**
     * Train the sentiment analyzer on labeled data.
     * 
     * @param texts List of training texts
     * @param labels List of sentiment labels (0.0 = negative, 1.0 = positive)
     * @param epochs Number of training epochs
     */
    public void train(List<String> texts, List<Double> labels, int epochs) {
        if (texts.size() != labels.size()) {
            throw new IllegalArgumentException("Texts and labels must have same size");
        }
        
        // Build vocabulary from all texts
        StringBuilder allText = new StringBuilder();
        for (String text : texts) {
            allText.append(text).append(" ");
        }
        vocabulary.buildFromText(allText.toString());
        
        // Initialize GRU
        int inputSize = vocabulary.getSize();
        gru = new GRUCell(inputSize, hiddenSize);
        
        // Initialize output layer
        outputWeights = new double[hiddenSize][1];
        outputBias = new double[1];
        
        Random random = new Random(42);
        for (int i = 0; i < hiddenSize; i++) {
            outputWeights[i][0] = random.nextGaussian() * 0.1;
        }
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                double target = labels.get(i);
                
                // Reset GRU state
                gru.resetState();
                
                // Process text character by character
                String[] characters = text.split("");
                for (String ch : characters) {
                    if (vocabulary.contains(ch)) {
                        double[] input = vocabulary.oneHotEncode(ch);
                        gru.forward(input);
                    }
                }
                
                // Get final hidden state and compute output
                double[] hiddenState = gru.getHiddenState();
                double output = computeOutput(hiddenState);
                
                // Compute loss (binary cross-entropy)
                double loss = binaryCrossEntropyLoss(output, target);
                totalLoss += loss;
                
                // Simplified backpropagation (for demonstration)
                // In a real implementation, you would compute gradients and update weights
            }
            
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d, Average Loss: %.4f\n", epoch, totalLoss / texts.size());
            }
        }
        
        isTrained = true;
    }
    
    /**
     * Analyze sentiment of a given text.
     * 
     * @param text Input text to analyze
     * @return Sentiment score (0.0 = negative, 1.0 = positive)
     */
    public double analyzeSentiment(String text) {
        if (isTrained && gru != null) {
            return analyzeWithRNN(text);
        } else {
            return analyzeWithLexicon(text);
        }
    }
    
    /**
     * Analyze sentiment using the trained RNN model.
     * 
     * @param text Input text
     * @return Sentiment score
     */
    private double analyzeWithRNN(String text) {
        gru.resetState();
        
        // Process text character by character
        String[] characters = text.split("");
        for (String ch : characters) {
            if (vocabulary.contains(ch)) {
                double[] input = vocabulary.oneHotEncode(ch);
                gru.forward(input);
            }
        }
        
        // Get final hidden state and compute output
        double[] hiddenState = gru.getHiddenState();
        return computeOutput(hiddenState);
    }
    
    /**
     * Analyze sentiment using lexicon-based approach (fallback).
     * 
     * @param text Input text
     * @return Sentiment score
     */
    private double analyzeWithLexicon(String text) {
        String[] words = text.toLowerCase().split("\\s+");
        double positiveScore = 0.0;
        double negativeScore = 0.0;
        
        for (String word : words) {
            // Remove punctuation
            word = word.replaceAll("[^a-zA-Z]", "");
            
            if (positiveWords.containsKey(word)) {
                positiveScore += positiveWords.get(word);
            }
            if (negativeWords.containsKey(word)) {
                negativeScore += Math.abs(negativeWords.get(word));
            }
        }
        
        // Normalize to [0, 1] range
        double totalScore = positiveScore + negativeScore;
        if (totalScore == 0) {
            return 0.5; // Neutral
        }
        
        return positiveScore / totalScore;
    }
    
    /**
     * Compute output from hidden state.
     * 
     * @param hiddenState GRU hidden state
     * @return Sentiment score
     */
    private double computeOutput(double[] hiddenState) {
        double output = outputBias[0];
        
        for (int i = 0; i < hiddenSize; i++) {
            output += outputWeights[i][0] * hiddenState[i];
        }
        
        // Apply sigmoid to get probability
        return sigmoid(output);
    }
    
    /**
     * Sigmoid activation function.
     * 
     * @param x Input value
     * @return Sigmoid output
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Compute binary cross-entropy loss.
     * 
     * @param predicted Predicted probability
     * @param target Target probability
     * @return Loss value
     */
    private double binaryCrossEntropyLoss(double predicted, double target) {
        return -target * Math.log(predicted + 1e-8) - (1 - target) * Math.log(1 - predicted + 1e-8);
    }
    
    /**
     * Get sentiment label from score.
     * 
     * @param score Sentiment score
     * @return Sentiment label
     */
    public String getSentimentLabel(double score) {
        if (score > 0.6) return "Positive";
        else if (score < 0.4) return "Negative";
        else return "Neutral";
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
     * Set hidden size for the GRU.
     * 
     * @param hiddenSize New hidden size
     */
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    /**
     * Check if model is trained.
     * 
     * @return True if model has been trained
     */
    public boolean isTrained() {
        return isTrained;
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
     * Add a positive word to the lexicon.
     * 
     * @param word Positive word
     */
    public void addPositiveWord(String word) {
        positiveWords.put(word.toLowerCase(), 1.0);
    }
    
    /**
     * Add a negative word to the lexicon.
     * 
     * @param word Negative word
     */
    public void addNegativeWord(String word) {
        negativeWords.put(word.toLowerCase(), -1.0);
    }
    
    /**
     * Get number of positive words in lexicon.
     * 
     * @return Number of positive words
     */
    public int getPositiveWordCount() {
        return positiveWords.size();
    }
    
    /**
     * Get number of negative words in lexicon.
     * 
     * @return Number of negative words
     */
    public int getNegativeWordCount() {
        return negativeWords.size();
    }
}

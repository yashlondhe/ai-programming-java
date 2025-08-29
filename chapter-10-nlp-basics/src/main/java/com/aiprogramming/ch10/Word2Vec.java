package com.aiprogramming.ch10;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Simplified Word2Vec implementation for word embeddings
 * Uses a neural network approach to learn word representations
 */
public class Word2Vec {
    
    private final Map<String, Integer> wordToIndex;
    private final List<String> indexToWord;
    private double[][] wordVectors;
    private final TextPreprocessor preprocessor;
    private final int vectorSize;
    private final int windowSize;
    private final double learningRate;
    private final Random random;
    
    public Word2Vec(int vectorSize, int windowSize, double learningRate) {
        this.vectorSize = vectorSize;
        this.windowSize = windowSize;
        this.learningRate = learningRate;
        this.wordToIndex = new HashMap<>();
        this.indexToWord = new ArrayList<>();
        this.preprocessor = new TextPreprocessor();
        this.random = new Random(42); // Fixed seed for reproducibility
        this.wordVectors = null; // Will be initialized after vocabulary is built
    }
    
    /**
     * Build vocabulary from documents
     */
    public void buildVocabulary(List<String> documents, int minFrequency) {
        Map<String, Integer> wordFrequency = preprocessor.getWordFrequency(documents, true, false);
        
        // Filter by minimum frequency
        wordFrequency = wordFrequency.entrySet().stream()
                .filter(entry -> entry.getValue() >= minFrequency)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        // Create word to index mapping
        int index = 0;
        for (String word : wordFrequency.keySet()) {
            wordToIndex.put(word, index);
            indexToWord.add(word);
            index++;
        }
    }
    
    /**
     * Initialize word vectors randomly
     */
    private void initializeWordVectors() {
        int vocabSize = wordToIndex.size();
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                wordVectors[i][j] = (random.nextDouble() - 0.5) * 0.1; // Small random values
            }
        }
    }
    
    /**
     * Generate training pairs (context, target) from documents
     */
    private List<Map.Entry<Integer, Integer>> generateTrainingPairs(List<String> documents) {
        List<Map.Entry<Integer, Integer>> pairs = new ArrayList<>();
        
        for (String document : documents) {
            List<String> tokens = preprocessor.preprocess(document, true, false);
            List<Integer> tokenIndices = new ArrayList<>();
            
            // Convert tokens to indices
            for (String token : tokens) {
                Integer index = wordToIndex.get(token);
                if (index != null) {
                    tokenIndices.add(index);
                }
            }
            
            // Generate context-target pairs
            for (int i = 0; i < tokenIndices.size(); i++) {
                int targetIndex = tokenIndices.get(i);
                
                // Generate context words within window
                for (int j = Math.max(0, i - windowSize); j < Math.min(tokenIndices.size(), i + windowSize + 1); j++) {
                    if (i != j) {
                        int contextIndex = tokenIndices.get(j);
                        pairs.add(new AbstractMap.SimpleEntry<>(contextIndex, targetIndex));
                    }
                }
            }
        }
        
        return pairs;
    }
    
    /**
     * Train Word2Vec model
     */
    public void train(List<String> documents, int epochs) {
        if (wordToIndex.isEmpty()) {
            throw new IllegalStateException("Vocabulary must be built before training");
        }
        
        int vocabSize = wordToIndex.size();
        double[][] vectors = new double[vocabSize][vectorSize];
        
        // Initialize vectors randomly
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                vectors[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }
        
        // Generate training pairs
        List<Map.Entry<Integer, Integer>> trainingPairs = generateTrainingPairs(documents);
        
        System.out.println("Training Word2Vec with " + trainingPairs.size() + " pairs over " + epochs + " epochs");
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (Map.Entry<Integer, Integer> pair : trainingPairs) {
                int contextIndex = pair.getKey();
                int targetIndex = pair.getValue();
                
                // Simple skip-gram training (simplified)
                double[] contextVector = vectors[contextIndex];
                double[] targetVector = vectors[targetIndex];
                
                // Compute similarity (dot product)
                double similarity = 0.0;
                for (int i = 0; i < vectorSize; i++) {
                    similarity += contextVector[i] * targetVector[i];
                }
                
                // Simple loss: we want high similarity for positive pairs
                double loss = -Math.log(sigmoid(similarity));
                totalLoss += loss;
                
                // Update vectors (simplified gradient descent)
                double gradient = sigmoid(similarity) - 1.0; // Target is 1 for positive pairs
                
                for (int i = 0; i < vectorSize; i++) {
                    double contextGrad = gradient * targetVector[i] * learningRate;
                    double targetGrad = gradient * contextVector[i] * learningRate;
                    
                    vectors[contextIndex][i] -= contextGrad;
                    vectors[targetIndex][i] -= targetGrad;
                }
            }
            
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d, Average Loss: %.4f%n", epoch, totalLoss / trainingPairs.size());
            }
        }
        
        // Store the trained vectors
        this.wordVectors = vectors;
    }
    
    /**
     * Sigmoid function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Get word vector for a given word
     */
    public double[] getWordVector(String word) {
        Integer index = wordToIndex.get(word);
        if (index == null || wordVectors == null) {
            return null;
        }
        return Arrays.copyOf(wordVectors[index], vectorSize);
    }
    
    /**
     * Find most similar words to a given word
     */
    public List<String> findMostSimilarWords(String word, int topK) {
        double[] queryVector = getWordVector(word);
        if (queryVector == null) {
            return new ArrayList<>();
        }
        
        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        
        for (Map.Entry<String, Integer> entry : wordToIndex.entrySet()) {
            String otherWord = entry.getKey();
            if (!otherWord.equals(word)) {
                double[] otherVector = wordVectors[entry.getValue()];
                double similarity = cosineSimilarity(queryVector, otherVector);
                similarities.add(new AbstractMap.SimpleEntry<>(otherWord, similarity));
            }
        }
        
        return similarities.stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }
    
    /**
     * Compute cosine similarity between two vectors
     */
    private double cosineSimilarity(double[] vector1, double[] vector2) {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }
        
        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * Get vocabulary size
     */
    public int getVocabularySize() {
        return wordToIndex.size();
    }
    
    /**
     * Get vector size
     */
    public int getVectorSize() {
        return vectorSize;
    }
    
    /**
     * Check if word is in vocabulary
     */
    public boolean containsWord(String word) {
        return wordToIndex.containsKey(word);
    }
    
    /**
     * Print word similarities for demonstration
     */
    public void printWordSimilarities(String word, int topK) {
        System.out.println("Most similar words to '" + word + "':");
        List<String> similarWords = findMostSimilarWords(word, topK);
        for (int i = 0; i < similarWords.size(); i++) {
            System.out.printf("  %d. %s%n", i + 1, similarWords.get(i));
        }
    }
}

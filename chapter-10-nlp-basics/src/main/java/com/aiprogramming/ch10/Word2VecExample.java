package com.aiprogramming.ch10;

import java.util.Arrays;
import java.util.List;

/**
 * Example demonstrating Word2Vec functionality
 */
public class Word2VecExample {
    
    public static void main(String[] args) {
        System.out.println("=== Word2Vec Example ===\n");
        
        // Create Word2Vec model
        Word2Vec word2vec = new Word2Vec(50, 2, 0.01); // 50-dimensional vectors, window size 2, learning rate 0.01
        
        // Sample documents for training
        List<String> trainingDocuments = Arrays.asList(
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Text preprocessing is essential for natural language processing tasks.",
            "Artificial intelligence encompasses machine learning and deep learning techniques.",
            "Neural networks are inspired by biological brain structures.",
            "Computer vision processes and analyzes visual information.",
            "Speech recognition converts spoken words into text.",
            "Data science combines statistics and machine learning.",
            "Big data refers to large and complex datasets.",
            "Supervised learning uses labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Reinforcement learning learns through trial and error.",
            "Feature engineering creates meaningful input representations.",
            "Model evaluation measures prediction accuracy and performance.",
            "Cross validation ensures robust model assessment.",
            "Overfitting occurs when models memorize training data.",
            "Regularization prevents overfitting in machine learning models.",
            "Gradient descent optimizes model parameters during training.",
            "Backpropagation computes gradients in neural networks."
        );
        
        System.out.println("Training Word2Vec model...");
        System.out.println("Number of documents: " + trainingDocuments.size());
        System.out.println("Vector size: " + word2vec.getVectorSize());
        System.out.println("Window size: " + 2);
        System.out.println();
        
        // Build vocabulary
        word2vec.buildVocabulary(trainingDocuments, 1); // Minimum frequency of 1
        System.out.println("Vocabulary size: " + word2vec.getVocabularySize());
        System.out.println();
        
        // Train the model
        System.out.println("Training model (this may take a moment)...");
        word2vec.train(trainingDocuments, 50); // 50 epochs
        System.out.println();
        
        // Test word similarities
        System.out.println("=== Word Similarity Examples ===");
        List<String> testWords = Arrays.asList(
            "machine", "learning", "artificial", "intelligence", "neural", "networks",
            "natural", "language", "processing", "deep", "data", "science"
        );
        
        for (String word : testWords) {
            if (word2vec.containsWord(word)) {
                System.out.println("Similar words to '" + word + "':");
                List<String> similarWords = word2vec.findMostSimilarWords(word, 5);
                for (int i = 0; i < similarWords.size(); i++) {
                    System.out.printf("  %d. %s%n", i + 1, similarWords.get(i));
                }
                System.out.println();
            }
        }
        
        // Word vector analysis
        System.out.println("=== Word Vector Analysis ===");
        String[] analysisWords = {"machine", "learning", "artificial", "intelligence"};
        
        for (String word : analysisWords) {
            if (word2vec.containsWord(word)) {
                double[] vector = word2vec.getWordVector(word);
                System.out.printf("Vector for '%s' (first 10 dimensions): [", word);
                for (int i = 0; i < Math.min(10, vector.length); i++) {
                    System.out.printf("%.4f", vector[i]);
                    if (i < Math.min(9, vector.length - 1)) {
                        System.out.print(", ");
                    }
                }
                if (vector.length > 10) {
                    System.out.print(", ...");
                }
                System.out.println("]");
            }
        }
        System.out.println();
        
        // Semantic relationships
        System.out.println("=== Semantic Relationships ===");
        
        // Test analogy-like relationships
        String[] analogyTests = {
            "machine", "learning", "artificial", "intelligence",
            "neural", "networks", "deep", "learning",
            "natural", "language", "text", "processing"
        };
        
        for (int i = 0; i < analogyTests.length; i += 4) {
            if (i + 3 < analogyTests.length) {
                String word1 = analogyTests[i];
                String word2 = analogyTests[i + 1];
                String word3 = analogyTests[i + 2];
                String word4 = analogyTests[i + 3];
                
                System.out.printf("Relationship: %s is to %s as %s is to %s%n", 
                    word1, word2, word3, word4);
                
                if (word2vec.containsWord(word1) && word2vec.containsWord(word2) && 
                    word2vec.containsWord(word3) && word2vec.containsWord(word4)) {
                    
                    double[] vec1 = word2vec.getWordVector(word1);
                    double[] vec2 = word2vec.getWordVector(word2);
                    double[] vec3 = word2vec.getWordVector(word3);
                    double[] vec4 = word2vec.getWordVector(word4);
                    
                    // Compute similarities
                    double sim12 = cosineSimilarity(vec1, vec2);
                    double sim34 = cosineSimilarity(vec3, vec4);
                    double sim13 = cosineSimilarity(vec1, vec3);
                    double sim24 = cosineSimilarity(vec2, vec4);
                    
                    System.out.printf("  Similarities: %s-%s: %.4f, %s-%s: %.4f%n", 
                        word1, word2, sim12, word3, word4, sim34);
                    System.out.printf("  Cross-similarities: %s-%s: %.4f, %s-%s: %.4f%n", 
                        word1, word3, sim13, word2, word4, sim24);
                }
                System.out.println();
            }
        }
        
        // Vocabulary exploration
        System.out.println("=== Vocabulary Exploration ===");
        System.out.println("Sample words in vocabulary:");
        
        // Get some sample words from the vocabulary
        List<String> sampleWords = Arrays.asList(
            "machine", "learning", "artificial", "intelligence", "neural", "networks",
            "natural", "language", "processing", "deep", "data", "science", "computer",
            "vision", "speech", "recognition", "supervised", "unsupervised", "reinforcement"
        );
        
        for (String word : sampleWords) {
            if (word2vec.containsWord(word)) {
                System.out.printf("  ✓ %s%n", word);
            } else {
                System.out.printf("  ✗ %s (not in vocabulary)%n", word);
            }
        }
        System.out.println();
        
        // Word clustering example
        System.out.println("=== Word Clustering Example ===");
        String[] techWords = {"machine", "learning", "artificial", "intelligence", "neural", "networks"};
        String[] processWords = {"processing", "analysis", "computation", "algorithm", "optimization"};
        
        System.out.println("Tech-related words:");
        for (String word : techWords) {
            if (word2vec.containsWord(word)) {
                List<String> similar = word2vec.findMostSimilarWords(word, 3);
                System.out.printf("  %s: %s%n", word, similar);
            }
        }
        System.out.println();
        
        System.out.println("Process-related words:");
        for (String word : processWords) {
            if (word2vec.containsWord(word)) {
                List<String> similar = word2vec.findMostSimilarWords(word, 3);
                System.out.printf("  %s: %s%n", word, similar);
            }
        }
        System.out.println();
        
        // Interactive word similarity
        System.out.println("=== Interactive Word Similarity ===");
        String[] interactiveWords = {"learning", "intelligence", "networks", "processing"};
        
        for (String word : interactiveWords) {
            word2vec.printWordSimilarities(word, 8);
        }
        
        System.out.println("=== Word2Vec Example Complete ===");
    }
    
    /**
     * Compute cosine similarity between two vectors
     */
    private static double cosineSimilarity(double[] vector1, double[] vector2) {
        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        
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
}

package com.aiprogramming.ch10;

import java.util.*;

/**
 * Term Frequency-Inverse Document Frequency (TF-IDF) implementation
 * Provides better text representation than simple bag of words
 */
public class TFIDF {
    
    private final Map<String, Integer> vocabulary;
    private final List<String> vocabularyList;
    private final Map<String, Double> idfScores;
    private final TextPreprocessor preprocessor;
    private int totalDocuments;
    
    public TFIDF() {
        this.vocabulary = new HashMap<>();
        this.vocabularyList = new ArrayList<>();
        this.idfScores = new HashMap<>();
        this.preprocessor = new TextPreprocessor();
        this.totalDocuments = 0;
    }
    
    /**
     * Build vocabulary and compute IDF scores from training documents
     */
    public void fit(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        this.totalDocuments = documents.size();
        
        // Build vocabulary
        Set<String> vocabSet = preprocessor.buildVocabulary(documents, removeStopWords, applyStemming);
        int index = 0;
        for (String word : vocabSet) {
            vocabulary.put(word, index);
            vocabularyList.add(word);
            index++;
        }
        
        // Compute document frequency for each word
        Map<String, Integer> documentFrequency = new HashMap<>();
        for (String document : documents) {
            List<String> tokens = preprocessor.preprocess(document, removeStopWords, applyStemming);
            Set<String> uniqueTokens = new HashSet<>(tokens);
            
            for (String token : uniqueTokens) {
                if (vocabulary.containsKey(token)) {
                    documentFrequency.put(token, documentFrequency.getOrDefault(token, 0) + 1);
                }
            }
        }
        
        // Compute IDF scores
        for (String word : vocabulary.keySet()) {
            int df = documentFrequency.getOrDefault(word, 0);
            double idf = Math.log((double) totalDocuments / (df + 1)); // Add 1 to avoid division by zero
            idfScores.put(word, idf);
        }
    }
    
    /**
     * Transform documents to TF-IDF vectors
     */
    public List<double[]> transform(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        List<double[]> vectors = new ArrayList<>();
        
        for (String document : documents) {
            double[] vector = transformDocument(document, removeStopWords, applyStemming);
            vectors.add(vector);
        }
        
        return vectors;
    }
    
    /**
     * Transform a single document to TF-IDF vector
     */
    public double[] transformDocument(String document, boolean removeStopWords, boolean applyStemming) {
        double[] vector = new double[vocabulary.size()];
        
        List<String> tokens = preprocessor.preprocess(document, removeStopWords, applyStemming);
        
        // Count term frequencies
        Map<String, Integer> termFreq = new HashMap<>();
        for (String token : tokens) {
            if (vocabulary.containsKey(token)) {
                termFreq.put(token, termFreq.getOrDefault(token, 0) + 1);
            }
        }
        
        // Compute TF-IDF scores
        int totalTerms = tokens.size();
        for (Map.Entry<String, Integer> entry : termFreq.entrySet()) {
            String word = entry.getKey();
            int tf = entry.getValue();
            Integer index = vocabulary.get(word);
            
            if (index != null) {
                // TF: term frequency / total terms in document
                double tfScore = (double) tf / totalTerms;
                // TF-IDF = TF * IDF
                double tfidfScore = tfScore * idfScores.get(word);
                vector[index] = tfidfScore;
            }
        }
        
        return vector;
    }
    
    /**
     * Fit and transform in one step
     */
    public List<double[]> fitTransform(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        fit(documents, removeStopWords, applyStemming);
        return transform(documents, removeStopWords, applyStemming);
    }
    
    /**
     * Get vocabulary size
     */
    public int getVocabularySize() {
        return vocabulary.size();
    }
    
    /**
     * Get vocabulary as list
     */
    public List<String> getVocabulary() {
        return new ArrayList<>(vocabularyList);
    }
    
    /**
     * Get IDF score for a word
     */
    public double getIDFScore(String word) {
        return idfScores.getOrDefault(word, 0.0);
    }
    
    /**
     * Get top words by IDF score (most distinctive words)
     */
    public List<String> getTopWordsByIDF(int topK) {
        return idfScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Get feature names (words) for each dimension
     */
    public String[] getFeatureNames() {
        return vocabularyList.toArray(new String[0]);
    }
    
    /**
     * Compute cosine similarity between two TF-IDF vectors
     */
    public double cosineSimilarity(double[] vector1, double[] vector2) {
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
    
    /**
     * Find most similar documents
     */
    public List<Integer> findMostSimilarDocuments(double[] queryVector, List<double[]> documentVectors, int topK) {
        List<Map.Entry<Integer, Double>> similarities = new ArrayList<>();
        
        for (int i = 0; i < documentVectors.size(); i++) {
            double similarity = cosineSimilarity(queryVector, documentVectors.get(i));
            similarities.add(new AbstractMap.SimpleEntry<>(i, similarity));
        }
        
        return similarities.stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(topK)
                .map(Map.Entry::getKey)
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Print TF-IDF statistics
     */
    public void printTFIDFStats() {
        System.out.println("TF-IDF Statistics:");
        System.out.println("Total documents: " + totalDocuments);
        System.out.println("Vocabulary size: " + vocabulary.size());
        System.out.println("Top 10 words by IDF:");
        
        List<String> topWords = getTopWordsByIDF(10);
        for (String word : topWords) {
            System.out.printf("  %s: %.4f%n", word, idfScores.get(word));
        }
    }
}

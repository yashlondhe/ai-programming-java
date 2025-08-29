package com.aiprogramming.ch10;

import java.util.*;

/**
 * Bag of Words representation for text documents
 * Converts text documents into numerical feature vectors
 */
public class BagOfWords {
    
    private final Map<String, Integer> vocabulary;
    private final List<String> vocabularyList;
    private final TextPreprocessor preprocessor;
    
    public BagOfWords() {
        this.vocabulary = new HashMap<>();
        this.vocabularyList = new ArrayList<>();
        this.preprocessor = new TextPreprocessor();
    }
    
    /**
     * Build vocabulary from training documents
     */
    public void fit(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        Set<String> vocabSet = preprocessor.buildVocabulary(documents, removeStopWords, applyStemming);
        
        // Convert to indexed vocabulary
        int index = 0;
        for (String word : vocabSet) {
            vocabulary.put(word, index);
            vocabularyList.add(word);
            index++;
        }
    }
    
    /**
     * Transform documents to bag of words vectors
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
     * Transform a single document to bag of words vector
     */
    public double[] transformDocument(String document, boolean removeStopWords, boolean applyStemming) {
        double[] vector = new double[vocabulary.size()];
        
        List<String> tokens = preprocessor.preprocess(document, removeStopWords, applyStemming);
        
        for (String token : tokens) {
            Integer index = vocabulary.get(token);
            if (index != null) {
                vector[index]++;
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
     * Get word index in vocabulary
     */
    public Integer getWordIndex(String word) {
        return vocabulary.get(word);
    }
    
    /**
     * Get word by index
     */
    public String getWordByIndex(int index) {
        if (index >= 0 && index < vocabularyList.size()) {
            return vocabularyList.get(index);
        }
        return null;
    }
    
    /**
     * Get most frequent words in a document
     */
    public List<String> getMostFrequentWords(String document, int topK, boolean removeStopWords, boolean applyStemming) {
        List<String> tokens = preprocessor.preprocess(document, removeStopWords, applyStemming);
        Map<String, Integer> frequency = new HashMap<>();
        
        for (String token : tokens) {
            frequency.put(token, frequency.getOrDefault(token, 0) + 1);
        }
        
        return frequency.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
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
     * Print vocabulary statistics
     */
    public void printVocabularyStats() {
        System.out.println("Vocabulary Statistics:");
        System.out.println("Total words: " + vocabulary.size());
        System.out.println("Sample words: " + vocabularyList.subList(0, Math.min(10, vocabularyList.size())));
    }
}

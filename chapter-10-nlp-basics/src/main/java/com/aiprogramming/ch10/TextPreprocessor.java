package com.aiprogramming.ch10;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Text preprocessing utilities for NLP tasks
 * Handles tokenization, stemming, lemmatization, and text cleaning
 */
public class TextPreprocessor {
    
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final Pattern PUNCTUATION_PATTERN = Pattern.compile("[^a-zA-Z0-9\\s]");
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\b\\d+\\b");
    
    // Common stop words in English
    private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
        "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
        "i", "you", "your", "we", "they", "them", "this", "these", "those", "but", "or",
        "if", "then", "else", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "can", "will", "just", "should", "now"
    ));
    
    /**
     * Clean text by removing extra whitespace, converting to lowercase
     */
    public String cleanText(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }
        
        // Convert to lowercase
        text = text.toLowerCase();
        
        // Remove extra whitespace
        text = WHITESPACE_PATTERN.matcher(text).replaceAll(" ");
        
        // Remove punctuation (optional - can be controlled by parameter)
        text = PUNCTUATION_PATTERN.matcher(text).replaceAll(" ");
        
        // Remove numbers (optional)
        text = NUMBER_PATTERN.matcher(text).replaceAll(" ");
        
        // Final whitespace cleanup
        text = WHITESPACE_PATTERN.matcher(text).replaceAll(" ").trim();
        
        return text;
    }
    
    /**
     * Tokenize text into words
     */
    public List<String> tokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        String cleanedText = cleanText(text);
        return Arrays.asList(cleanedText.split("\\s+"));
    }
    
    /**
     * Remove stop words from tokenized text
     */
    public List<String> removeStopWords(List<String> tokens) {
        return tokens.stream()
                .filter(token -> !STOP_WORDS.contains(token.toLowerCase()))
                .filter(token -> token.length() > 1) // Remove single characters
                .collect(Collectors.toList());
    }
    
    /**
     * Simple stemming using Porter Stemmer algorithm (simplified version)
     */
    public String stem(String word) {
        if (word == null || word.length() < 3) {
            return word;
        }
        
        // Simple stemming rules (simplified Porter algorithm)
        String stemmed = word;
        
        // Remove common suffixes
        if (stemmed.endsWith("ing")) {
            stemmed = stemmed.substring(0, stemmed.length() - 3);
        } else if (stemmed.endsWith("ed")) {
            stemmed = stemmed.substring(0, stemmed.length() - 2);
        } else if (stemmed.endsWith("er")) {
            stemmed = stemmed.substring(0, stemmed.length() - 2);
        } else if (stemmed.endsWith("est")) {
            stemmed = stemmed.substring(0, stemmed.length() - 3);
        } else if (stemmed.endsWith("ly")) {
            stemmed = stemmed.substring(0, stemmed.length() - 2);
        } else if (stemmed.endsWith("s")) {
            stemmed = stemmed.substring(0, stemmed.length() - 1);
        }
        
        return stemmed;
    }
    
    /**
     * Apply stemming to all tokens
     */
    public List<String> stemTokens(List<String> tokens) {
        return tokens.stream()
                .map(this::stem)
                .collect(Collectors.toList());
    }
    
    /**
     * Complete text preprocessing pipeline
     */
    public List<String> preprocess(String text, boolean removeStopWords, boolean applyStemming) {
        List<String> tokens = tokenize(text);
        
        if (removeStopWords) {
            tokens = removeStopWords(tokens);
        }
        
        if (applyStemming) {
            tokens = stemTokens(tokens);
        }
        
        return tokens;
    }
    
    /**
     * Build vocabulary from a collection of documents
     */
    public Set<String> buildVocabulary(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        Set<String> vocabulary = new HashSet<>();
        
        for (String document : documents) {
            List<String> tokens = preprocess(document, removeStopWords, applyStemming);
            vocabulary.addAll(tokens);
        }
        
        return vocabulary;
    }
    
    /**
     * Create n-grams from tokens
     */
    public List<String> createNGrams(List<String> tokens, int n) {
        List<String> ngrams = new ArrayList<>();
        
        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder ngram = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) ngram.append(" ");
                ngram.append(tokens.get(i + j));
            }
            ngrams.add(ngram.toString());
        }
        
        return ngrams;
    }
    
    /**
     * Extract sentences from text (simple implementation)
     */
    public List<String> extractSentences(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // Simple sentence splitting by common sentence endings
        String[] sentences = text.split("[.!?]+\\s+");
        return Arrays.asList(sentences);
    }
    
    /**
     * Get word frequency from a collection of documents
     */
    public Map<String, Integer> getWordFrequency(List<String> documents, boolean removeStopWords, boolean applyStemming) {
        Map<String, Integer> frequency = new HashMap<>();
        
        for (String document : documents) {
            List<String> tokens = preprocess(document, removeStopWords, applyStemming);
            for (String token : tokens) {
                frequency.put(token, frequency.getOrDefault(token, 0) + 1);
            }
        }
        
        return frequency;
    }
}

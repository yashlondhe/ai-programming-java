package com.aiprogramming.ch10;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.ArrayList;

/**
 * Example demonstrating text preprocessing functionality
 */
public class TextPreprocessingExample {
    
    public static void main(String[] args) {
        System.out.println("=== Text Preprocessing Example ===\n");
        
        TextPreprocessor preprocessor = new TextPreprocessor();
        
        // Sample text
        String sampleText = "The quick brown fox jumps over the lazy dog. " +
                           "Natural language processing is an exciting field of artificial intelligence!";
        
        System.out.println("Original text:");
        System.out.println(sampleText);
        System.out.println();
        
        // Clean text
        String cleanedText = preprocessor.cleanText(sampleText);
        System.out.println("Cleaned text:");
        System.out.println(cleanedText);
        System.out.println();
        
        // Tokenize
        List<String> tokens = preprocessor.tokenize(sampleText);
        System.out.println("Tokenized text:");
        System.out.println(tokens);
        System.out.println();
        
        // Remove stop words
        List<String> tokensWithoutStopWords = preprocessor.removeStopWords(tokens);
        System.out.println("Tokens without stop words:");
        System.out.println(tokensWithoutStopWords);
        System.out.println();
        
        // Apply stemming
        List<String> stemmedTokens = preprocessor.stemTokens(tokensWithoutStopWords);
        System.out.println("Stemmed tokens:");
        System.out.println(stemmedTokens);
        System.out.println();
        
        // Complete preprocessing pipeline
        List<String> processedTokens = preprocessor.preprocess(sampleText, true, true);
        System.out.println("Complete preprocessing pipeline (remove stop words + stemming):");
        System.out.println(processedTokens);
        System.out.println();
        
        // Extract sentences
        List<String> sentences = preprocessor.extractSentences(sampleText);
        System.out.println("Extracted sentences:");
        for (int i = 0; i < sentences.size(); i++) {
            System.out.println((i + 1) + ". " + sentences.get(i));
        }
        System.out.println();
        
        // Create n-grams
        List<String> bigrams = preprocessor.createNGrams(tokens, 2);
        System.out.println("Bigrams:");
        System.out.println(bigrams);
        System.out.println();
        
        List<String> trigrams = preprocessor.createNGrams(tokens, 3);
        System.out.println("Trigrams:");
        System.out.println(trigrams);
        System.out.println();
        
        // Multiple documents example
        List<String> documents = Arrays.asList(
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Text preprocessing is essential for NLP tasks."
        );
        
        System.out.println("=== Multiple Documents Example ===");
        System.out.println("Documents:");
        for (int i = 0; i < documents.size(); i++) {
            System.out.println((i + 1) + ". " + documents.get(i));
        }
        System.out.println();
        
        // Build vocabulary
        Set<String> vocab = preprocessor.buildVocabulary(documents, true, false);
        System.out.println("Vocabulary (without stop words, no stemming):");
        System.out.println("Size: " + vocab.size());
        System.out.println("Words: " + new ArrayList<>(vocab));
        System.out.println();
        
        // Word frequency
        Set<String> vocabWithStemming = preprocessor.buildVocabulary(documents, true, true);
        System.out.println("Vocabulary (without stop words, with stemming):");
        System.out.println("Size: " + vocabWithStemming.size());
        System.out.println("Words: " + new ArrayList<>(vocabWithStemming));
        System.out.println();
        
        // Word frequency analysis
        System.out.println("=== Word Frequency Analysis ===");
        for (String doc : documents) {
            System.out.println("Document: " + doc);
            List<String> docTokens = preprocessor.preprocess(doc, true, false);
            System.out.println("Processed tokens: " + docTokens);
            System.out.println();
        }
        
        System.out.println("=== Text Preprocessing Complete ===");
    }
}

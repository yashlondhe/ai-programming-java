# Chapter 10: Text Processing and Analysis

## Introduction

Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. Text processing and analysis form the foundation of NLP, providing the essential tools and techniques needed to work with textual data effectively.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of text preprocessing and analysis
- Implement text preprocessing pipelines including tokenization, stemming, and stop word removal
- Create document vectorization using Bag of Words and TF-IDF representations
- Build word embeddings using simplified Word2Vec implementation
- Perform sentiment analysis using naive Bayes classification
- Extract named entities using pattern matching and rule-based approaches
- Apply text similarity measures and document matching techniques
- Build complete NLP applications for real-world problems

### Key Concepts

- **Text Preprocessing**: Cleaning and normalizing text data for analysis
- **Tokenization**: Breaking text into individual words or tokens
- **Document Vectorization**: Converting text documents into numerical representations
- **Word Embeddings**: Dense vector representations of words capturing semantic meaning
- **Sentiment Analysis**: Determining the emotional tone or opinion expressed in text
- **Named Entity Recognition**: Identifying and classifying named entities in text
- **Text Similarity**: Measuring the similarity between documents or words

## 10.1 Text Preprocessing

Text preprocessing is the first and crucial step in any NLP pipeline. Raw text data often contains noise, inconsistencies, and irrelevant information that must be cleaned and normalized before analysis.

### 10.1.1 Text Cleaning

Text cleaning involves removing unwanted elements and standardizing the text format.

#### Implementation

```java
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
}
```

### 10.1.2 Stemming

Stemming reduces words to their root form by removing common suffixes.

```java
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
```

### 10.1.3 Complete Preprocessing Pipeline

```java
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
```

## 10.2 Document Vectorization

Document vectorization converts text documents into numerical representations that can be processed by machine learning algorithms.

### 10.2.1 Bag of Words

The Bag of Words (BoW) model represents documents as vectors of word frequencies.

#### Implementation

```java
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
}
```

### 10.2.2 TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) provides better document representation by weighting words based on their importance.

#### Implementation

```java
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
}
```

## 10.3 Word Embeddings

Word embeddings represent words as dense vectors in a continuous vector space, capturing semantic relationships between words.

### 10.3.1 Word2Vec Implementation

Word2Vec learns word embeddings by predicting context words given a target word (skip-gram) or vice versa (CBOW).

#### Implementation

```java
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
}
```

## 10.4 Sentiment Analysis

Sentiment analysis determines the emotional tone or opinion expressed in text, classifying it as positive, negative, or neutral.

### 10.4.1 Naive Bayes Sentiment Classifier

```java
package com.aiprogramming.ch10;

import java.util.*;

/**
 * Simple sentiment analysis classifier
 * Uses TF-IDF features and a naive Bayes approach
 */
public class SentimentAnalyzer {
    
    private final TFIDF tfidf;
    private final TextPreprocessor preprocessor;
    private final Map<String, Double> positiveWordScores;
    private final Map<String, Double> negativeWordScores;
    private double positivePrior;
    private double negativePrior;
    
    public SentimentAnalyzer() {
        this.tfidf = new TFIDF();
        this.preprocessor = new TextPreprocessor();
        this.positiveWordScores = new HashMap<>();
        this.negativeWordScores = new HashMap<>();
        this.positivePrior = 0.0;
        this.negativePrior = 0.0;
    }
    
    /**
     * Train the sentiment analyzer
     * @param positiveTexts List of positive sentiment texts
     * @param negativeTexts List of negative sentiment texts
     */
    public void train(List<String> positiveTexts, List<String> negativeTexts) {
        // Combine all texts for TF-IDF training
        List<String> allTexts = new ArrayList<>();
        allTexts.addAll(positiveTexts);
        allTexts.addAll(negativeTexts);
        
        // Train TF-IDF
        tfidf.fit(allTexts, true, false);
        
        // Calculate priors
        int totalTexts = positiveTexts.size() + negativeTexts.size();
        double positivePrior = (double) positiveTexts.size() / totalTexts;
        double negativePrior = (double) negativeTexts.size() / totalTexts;
        
        // Calculate word scores for each class
        Map<String, Integer> positiveWordFreq = preprocessor.getWordFrequency(positiveTexts, true, false);
        Map<String, Integer> negativeWordFreq = preprocessor.getWordFrequency(negativeTexts, true, false);
        
        // Calculate total word counts
        int totalPositiveWords = positiveWordFreq.values().stream().mapToInt(Integer::intValue).sum();
        int totalNegativeWords = negativeWordFreq.values().stream().mapToInt(Integer::intValue).sum();
        
        // Calculate word probabilities with smoothing
        double smoothing = 1.0; // Add-1 smoothing
        int vocabSize = tfidf.getVocabularySize();
        
        for (String word : tfidf.getVocabulary()) {
            int positiveCount = positiveWordFreq.getOrDefault(word, 0);
            int negativeCount = negativeWordFreq.getOrDefault(word, 0);
            
            // P(word|positive) = (count + smoothing) / (total + vocab_size * smoothing)
            double positiveProb = (positiveCount + smoothing) / (totalPositiveWords + vocabSize * smoothing);
            double negativeProb = (negativeCount + smoothing) / (totalNegativeWords + vocabSize * smoothing);
            
            positiveWordScores.put(word, Math.log(positiveProb));
            negativeWordScores.put(word, Math.log(negativeProb));
        }
        
        this.positivePrior = Math.log(positivePrior);
        this.negativePrior = Math.log(negativePrior);
    }
    
    /**
     * Predict sentiment of a text
     * @param text Input text
     * @return 1.0 for positive, 0.0 for negative
     */
    public double predictSentiment(String text) {
        List<String> tokens = preprocessor.preprocess(text, true, false);
        
        double positiveScore = positivePrior;
        double negativeScore = negativePrior;
        
        for (String token : tokens) {
            if (positiveWordScores.containsKey(token)) {
                positiveScore += positiveWordScores.get(token);
                negativeScore += negativeWordScores.get(token);
            }
        }
        
        // Return probability of positive sentiment
        double positiveProb = 1.0 / (1.0 + Math.exp(negativeScore - positiveScore));
        return positiveProb;
    }
    
    /**
     * Predict sentiment class
     * @param text Input text
     * @return "positive" or "negative"
     */
    public String predictSentimentClass(String text) {
        double sentiment = predictSentiment(text);
        return sentiment > 0.5 ? "positive" : "negative";
    }
}
```

## 10.5 Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies named entities in text, such as person names, organizations, locations, dates, and monetary values.

### 10.5.1 Pattern-Based NER

```java
package com.aiprogramming.ch10;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Named Entity Recognition (NER) implementation
 * Uses pattern matching and rule-based approaches to identify entities
 */
public class NamedEntityRecognizer {
    
    private final TextPreprocessor preprocessor;
    private final Map<String, List<String>> entityPatterns;
    private final Map<String, Pattern> compiledPatterns;
    
    // Entity types
    public static final String PERSON = "PERSON";
    public static final String ORGANIZATION = "ORGANIZATION";
    public static final String LOCATION = "LOCATION";
    public static final String DATE = "DATE";
    public static final String MONEY = "MONEY";
    public static final String PERCENT = "PERCENT";
    public static final String TIME = "TIME";
    
    public NamedEntityRecognizer() {
        this.preprocessor = new TextPreprocessor();
        this.entityPatterns = new HashMap<>();
        this.compiledPatterns = new HashMap<>();
        initializePatterns();
    }
    
    /**
     * Initialize entity recognition patterns
     */
    private void initializePatterns() {
        // Person patterns (simplified)
        List<String> personPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+ [A-Z][a-z]+\\b",  // First Last
            "\\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\\b",  // First Middle Last
            "\\b[A-Z][a-z]+ [A-Z]\\.[A-Z][a-z]+\\b"  // First M. Last
        );
        entityPatterns.put(PERSON, personPatterns);
        
        // Organization patterns
        List<String> orgPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation|Organization)\\b",
            "\\b[A-Z][a-z]+ [A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation|Organization)\\b",
            "\\b[A-Z]+\\b"  // All caps words (like IBM, NASA)
        );
        entityPatterns.put(ORGANIZATION, orgPatterns);
        
        // Location patterns
        List<String> locationPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+, [A-Z]{2}\\b",  // City, State
            "\\b[A-Z][a-z]+ [A-Z][a-z]+, [A-Z]{2}\\b",  // City Name, State
            "\\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard|Drive|Lane)\\b",
            "\\b[A-Z][a-z]+ (City|Town|Village|County|State|Country)\\b"
        );
        entityPatterns.put(LOCATION, locationPatterns);
        
        // Date patterns
        List<String> datePatterns = Arrays.asList(
            "\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b",  // MM/DD/YYYY
            "\\b\\d{4}-\\d{1,2}-\\d{1,2}\\b",  // YYYY-MM-DD
            "\\b(January|February|March|April|May|June|July|August|September|October|November|December) \\d{1,2},? \\d{4}\\b",
            "\\b\\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \\d{4}\\b"
        );
        entityPatterns.put(DATE, datePatterns);
        
        // Money patterns
        List<String> moneyPatterns = Arrays.asList(
            "\\$\\d+(\\.\\d{2})?\\b",
            "\\b\\d+(\\.\\d{2})? (dollars|USD)\\b",
            "\\b\\d+ (cents|pennies)\\b"
        );
        entityPatterns.put(MONEY, moneyPatterns);
        
        // Percentage patterns
        List<String> percentPatterns = Arrays.asList(
            "\\b\\d+(\\.\\d+)?%\\b",
            "\\b\\d+(\\.\\d+)? percent\\b"
        );
        entityPatterns.put(PERCENT, percentPatterns);
        
        // Time patterns
        List<String> timePatterns = Arrays.asList(
            "\\b\\d{1,2}:\\d{2} (AM|PM)\\b",
            "\\b\\d{1,2}:\\d{2}\\b",
            "\\b(noon|midnight)\\b"
        );
        entityPatterns.put(TIME, timePatterns);
        
        // Compile all patterns
        for (Map.Entry<String, List<String>> entry : entityPatterns.entrySet()) {
            for (String pattern : entry.getValue()) {
                compiledPatterns.put(pattern, Pattern.compile(pattern));
            }
        }
    }
    
    /**
     * Extract named entities from text
     * @param text Input text
     * @return Map of entity type to list of entities found
     */
    public Map<String, List<String>> extractEntities(String text) {
        Map<String, List<String>> entities = new HashMap<>();
        
        // Initialize entity lists
        for (String entityType : entityPatterns.keySet()) {
            entities.put(entityType, new ArrayList<>());
        }
        
        // Extract entities for each type
        for (Map.Entry<String, List<String>> entry : entityPatterns.entrySet()) {
            String entityType = entry.getKey();
            List<String> patterns = entry.getValue();
            
            for (String pattern : patterns) {
                Pattern compiledPattern = compiledPatterns.get(pattern);
                java.util.regex.Matcher matcher = compiledPattern.matcher(text);
                
                while (matcher.find()) {
                    String entity = matcher.group();
                    if (!entities.get(entityType).contains(entity)) {
                        entities.get(entityType).add(entity);
                    }
                }
            }
        }
        
        return entities;
    }
}
```

## 10.6 Text Similarity and Document Matching

Text similarity measures help identify how similar two documents or words are, enabling applications like document clustering, information retrieval, and plagiarism detection.

### 10.6.1 Cosine Similarity

```java
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
```

## 10.7 Practical Applications

### 10.7.1 Document Classification

```java
// Example: Classifying news articles into categories
public class DocumentClassifier {
    private final TFIDF tfidf;
    private final Map<String, double[]> categoryVectors;
    
    public DocumentClassifier() {
        this.tfidf = new TFIDF();
        this.categoryVectors = new HashMap<>();
    }
    
    public void train(Map<String, List<String>> trainingData) {
        // Combine all documents for TF-IDF training
        List<String> allDocuments = new ArrayList<>();
        for (List<String> documents : trainingData.values()) {
            allDocuments.addAll(documents);
        }
        
        // Train TF-IDF
        tfidf.fit(allDocuments, true, false);
        
        // Create category vectors
        for (Map.Entry<String, List<String>> entry : trainingData.entrySet()) {
            String category = entry.getKey();
            List<String> documents = entry.getValue();
            
            // Average TF-IDF vectors for documents in this category
            List<double[]> vectors = tfidf.transform(documents, true, false);
            double[] avgVector = new double[tfidf.getVocabularySize()];
            
            for (double[] vector : vectors) {
                for (int i = 0; i < vector.length; i++) {
                    avgVector[i] += vector[i];
                }
            }
            
            // Normalize
            for (int i = 0; i < avgVector.length; i++) {
                avgVector[i] /= vectors.size();
            }
            
            categoryVectors.put(category, avgVector);
        }
    }
    
    public String classify(String document) {
        double[] docVector = tfidf.transformDocument(document, true, false);
        
        String bestCategory = null;
        double bestSimilarity = -1.0;
        
        for (Map.Entry<String, double[]> entry : categoryVectors.entrySet()) {
            String category = entry.getKey();
            double[] categoryVector = entry.getValue();
            
            double similarity = cosineSimilarity(docVector, categoryVector);
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestCategory = category;
            }
        }
        
        return bestCategory;
    }
}
```

### 10.7.2 Information Retrieval System

```java
// Example: Simple search engine
public class InformationRetrievalSystem {
    private final TFIDF tfidf;
    private final List<String> documents;
    private final List<double[]> documentVectors;
    
    public InformationRetrievalSystem(List<String> documents) {
        this.tfidf = new TFIDF();
        this.documents = documents;
        
        // Train TF-IDF and transform documents
        this.documentVectors = tfidf.fitTransform(documents, true, false);
    }
    
    public List<SearchResult> search(String query, int topK) {
        double[] queryVector = tfidf.transformDocument(query, true, false);
        
        List<SearchResult> results = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            double similarity = tfidf.cosineSimilarity(queryVector, documentVectors.get(i));
            results.add(new SearchResult(i, documents.get(i), similarity));
        }
        
        // Sort by similarity and return top K
        return results.stream()
                .sorted(Comparator.comparing(SearchResult::getSimilarity).reversed())
                .limit(topK)
                .collect(Collectors.toList());
    }
    
    public static class SearchResult {
        private final int documentId;
        private final String document;
        private final double similarity;
        
        public SearchResult(int documentId, String document, double similarity) {
            this.documentId = documentId;
            this.document = document;
            this.similarity = similarity;
        }
        
        // Getters
        public int getDocumentId() { return documentId; }
        public String getDocument() { return document; }
        public double getSimilarity() { return similarity; }
    }
}
```

## 10.8 Performance Considerations

### 10.8.1 Memory Optimization

- **Sparse Vectors**: Use sparse representations for high-dimensional vectors
- **Vocabulary Pruning**: Remove low-frequency terms to reduce memory usage
- **Batch Processing**: Process documents in batches for large datasets

### 10.8.2 Computational Efficiency

- **Caching**: Cache compiled regex patterns and precomputed values
- **Vectorization**: Use efficient vector operations and libraries
- **Parallel Processing**: Leverage multi-threading for independent operations

### 10.8.3 Scalability

- **Incremental Learning**: Support for adding new documents without retraining
- **Distributed Processing**: Scale across multiple machines for large datasets
- **Streaming**: Process text data in real-time streams

## 10.9 Best Practices

### 10.9.1 Text Preprocessing

1. **Consistency**: Apply the same preprocessing to training and test data
2. **Domain-Specific**: Adapt preprocessing for your specific domain
3. **Validation**: Validate preprocessing steps on sample data
4. **Documentation**: Document preprocessing choices and their rationale

### 10.9.2 Feature Engineering

1. **Vocabulary Size**: Balance vocabulary size with computational efficiency
2. **Feature Selection**: Remove irrelevant or redundant features
3. **Normalization**: Apply appropriate normalization for your use case
4. **Dimensionality Reduction**: Consider techniques like PCA for high-dimensional data

### 10.9.3 Model Evaluation

1. **Cross-Validation**: Use cross-validation for robust evaluation
2. **Multiple Metrics**: Evaluate using multiple relevant metrics
3. **Error Analysis**: Analyze errors to understand model limitations
4. **Domain Expertise**: Incorporate domain knowledge in evaluation

## 10.10 Advanced Topics

### 10.10.1 Advanced Preprocessing

- **Lemmatization**: More sophisticated word form reduction
- **Part-of-Speech Tagging**: Identify grammatical parts of speech
- **Dependency Parsing**: Analyze syntactic relationships
- **Coreference Resolution**: Resolve pronoun references

### 10.10.2 Advanced Vectorization

- **Word Embeddings**: Pre-trained embeddings like Word2Vec, GloVe, BERT
- **Document Embeddings**: Doc2Vec, sentence transformers
- **Contextual Embeddings**: BERT, GPT, and transformer-based embeddings
- **Multilingual Embeddings**: Support for multiple languages

### 10.10.3 Advanced NLP Tasks

- **Machine Translation**: Translate text between languages
- **Question Answering**: Answer questions based on text passages
- **Text Summarization**: Generate concise summaries of documents
- **Dialogue Systems**: Build conversational AI systems

## 10.11 Summary

This chapter has covered the fundamental concepts and techniques of text processing and analysis in Java. We've implemented:

- **Text Preprocessing**: Cleaning, tokenization, stemming, and stop word removal
- **Document Vectorization**: Bag of Words and TF-IDF representations
- **Word Embeddings**: Simplified Word2Vec implementation
- **Sentiment Analysis**: Naive Bayes-based sentiment classification
- **Named Entity Recognition**: Pattern-based entity extraction
- **Text Similarity**: Cosine similarity and document matching
- **Practical Applications**: Document classification and information retrieval

These techniques form the foundation for more advanced NLP applications. The modular design allows for easy extension and customization for specific use cases. The comprehensive examples demonstrate practical applications of each technique, making it easy to understand and apply these concepts to real-world problems.

### Key Takeaways

1. **Text preprocessing is crucial**: Proper preprocessing significantly impacts NLP model performance
2. **Choose appropriate representations**: Different vectorization techniques suit different applications
3. **Word embeddings capture semantics**: Dense vector representations enable semantic similarity
4. **Pattern-based approaches work well**: Rule-based methods can be effective for structured tasks
5. **Evaluation is essential**: Use appropriate metrics and validation techniques
6. **Scalability matters**: Consider performance and scalability for production systems

### Next Steps

The next chapter will explore advanced NLP techniques with transformers, including BERT and other modern language models. These advanced techniques build upon the foundations established in this chapter and enable more sophisticated language understanding and generation capabilities.

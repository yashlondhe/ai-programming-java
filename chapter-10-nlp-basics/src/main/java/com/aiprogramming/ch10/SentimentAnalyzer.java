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
    
    /**
     * Get sentiment confidence
     * @param text Input text
     * @return Confidence score (0.0 to 1.0)
     */
    public double getSentimentConfidence(String text) {
        double sentiment = predictSentiment(text);
        return Math.abs(sentiment - 0.5) * 2; // Convert to 0-1 confidence
    }
    
    /**
     * Get most indicative words for each sentiment
     * @param topK Number of top words to return
     * @return Map with "positive" and "negative" lists of words
     */
    public Map<String, List<String>> getMostIndicativeWords(int topK) {
        Map<String, List<String>> result = new HashMap<>();
        
        // Calculate word importance (difference in log probabilities)
        List<Map.Entry<String, Double>> wordImportance = new ArrayList<>();
        for (String word : tfidf.getVocabulary()) {
            double positiveScore = positiveWordScores.get(word);
            double negativeScore = negativeWordScores.get(word);
            double importance = positiveScore - negativeScore;
            wordImportance.add(new AbstractMap.SimpleEntry<>(word, importance));
        }
        
        // Sort by importance
        wordImportance.sort(Map.Entry.<String, Double>comparingByValue().reversed());
        
        // Get top positive and negative words
        List<String> positiveWords = new ArrayList<>();
        List<String> negativeWords = new ArrayList<>();
        
        for (int i = 0; i < Math.min(topK, wordImportance.size()); i++) {
            Map.Entry<String, Double> entry = wordImportance.get(i);
            if (entry.getValue() > 0) {
                positiveWords.add(entry.getKey());
            }
        }
        
        for (int i = wordImportance.size() - 1; i >= Math.max(0, wordImportance.size() - topK); i--) {
            Map.Entry<String, Double> entry = wordImportance.get(i);
            if (entry.getValue() < 0) {
                negativeWords.add(entry.getKey());
            }
        }
        
        result.put("positive", positiveWords);
        result.put("negative", negativeWords);
        
        return result;
    }
    
    /**
     * Evaluate the model on test data
     * @param testTexts Test texts
     * @param testLabels True labels (1.0 for positive, 0.0 for negative)
     * @return Map with accuracy, precision, recall, and F1 score
     */
    public Map<String, Double> evaluate(List<String> testTexts, List<Double> testLabels) {
        if (testTexts.size() != testLabels.size()) {
            throw new IllegalArgumentException("Test texts and labels must have the same size");
        }
        
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < testTexts.size(); i++) {
            String text = testTexts.get(i);
            double trueLabel = testLabels.get(i);
            double predictedSentiment = predictSentiment(text);
            String predictedClass = predictedSentiment > 0.5 ? "positive" : "negative";
            String trueClass = trueLabel > 0.5 ? "positive" : "negative";
            
            if (predictedClass.equals("positive") && trueClass.equals("positive")) {
                truePositives++;
            } else if (predictedClass.equals("negative") && trueClass.equals("negative")) {
                trueNegatives++;
            } else if (predictedClass.equals("positive") && trueClass.equals("negative")) {
                falsePositives++;
            } else {
                falseNegatives++;
            }
        }
        
        // Calculate metrics
        double accuracy = (double) (truePositives + trueNegatives) / testTexts.size();
        double precision = truePositives == 0 ? 0.0 : (double) truePositives / (truePositives + falsePositives);
        double recall = truePositives == 0 ? 0.0 : (double) truePositives / (truePositives + falseNegatives);
        double f1Score = (precision + recall == 0) ? 0.0 : 2 * precision * recall / (precision + recall);
        
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("accuracy", accuracy);
        metrics.put("precision", precision);
        metrics.put("recall", recall);
        metrics.put("f1_score", f1Score);
        
        return metrics;
    }
    
    /**
     * Print model statistics
     */
    public void printModelStats() {
        System.out.println("Sentiment Analyzer Statistics:");
        System.out.println("Vocabulary size: " + tfidf.getVocabularySize());
        System.out.println("Positive prior: " + Math.exp(positivePrior));
        System.out.println("Negative prior: " + Math.exp(negativePrior));
        
        Map<String, List<String>> indicativeWords = getMostIndicativeWords(10);
        System.out.println("Top positive words: " + indicativeWords.get("positive"));
        System.out.println("Top negative words: " + indicativeWords.get("negative"));
    }
}

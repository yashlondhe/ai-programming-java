package com.aiprogramming.ch10;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Example demonstrating sentiment analysis functionality
 */
public class SentimentAnalysisExample {
    
    public static void main(String[] args) {
        System.out.println("=== Sentiment Analysis Example ===\n");
        
        SentimentAnalyzer analyzer = new SentimentAnalyzer();
        
        // Training data
        List<String> positiveTexts = Arrays.asList(
            "I love this product! It's amazing and works perfectly.",
            "Great service and excellent quality. Highly recommended!",
            "This is the best purchase I've ever made. Fantastic!",
            "Wonderful experience with this company. Very satisfied.",
            "Outstanding performance and great value for money.",
            "Excellent customer support and fast delivery.",
            "This exceeded my expectations. Absolutely brilliant!",
            "Perfect product with amazing features.",
            "Highly satisfied with the quality and service.",
            "Outstanding value and great user experience."
        );
        
        List<String> negativeTexts = Arrays.asList(
            "Terrible product. Complete waste of money.",
            "Awful customer service. Never buying again.",
            "This is the worst purchase I've ever made.",
            "Poor quality and disappointing performance.",
            "Bad experience with this company. Not recommended.",
            "Horrible product that broke immediately.",
            "Terrible customer support and slow delivery.",
            "This failed to meet my expectations. Very disappointed.",
            "Poor quality and expensive for what you get.",
            "Bad value and terrible user experience."
        );
        
        System.out.println("Training sentiment analyzer...");
        System.out.println("Positive examples: " + positiveTexts.size());
        System.out.println("Negative examples: " + negativeTexts.size());
        System.out.println();
        
        // Train the model
        analyzer.train(positiveTexts, negativeTexts);
        
        // Print model statistics
        analyzer.printModelStats();
        System.out.println();
        
        // Test texts
        List<String> testTexts = Arrays.asList(
            "This product is really good and I'm happy with it.",
            "Terrible quality, I hate this product.",
            "The service was okay, nothing special.",
            "Amazing experience, highly recommend!",
            "Disappointing results, not worth the money.",
            "Great value and excellent performance.",
            "Poor customer service and bad quality.",
            "Fantastic product with wonderful features.",
            "Mediocre performance, could be better.",
            "Outstanding quality and great service!"
        );
        
        System.out.println("=== Sentiment Analysis Results ===");
        for (int i = 0; i < testTexts.size(); i++) {
            String text = testTexts.get(i);
            double sentiment = analyzer.predictSentiment(text);
            String sentimentClass = analyzer.predictSentimentClass(text);
            double confidence = analyzer.getSentimentConfidence(text);
            
            System.out.printf("Text %d: %s%n", i + 1, text);
            System.out.printf("  Sentiment: %.4f (%s)%n", sentiment, sentimentClass);
            System.out.printf("  Confidence: %.4f%n", confidence);
            System.out.println();
        }
        
        // Get most indicative words
        System.out.println("=== Most Indicative Words ===");
        Map<String, List<String>> indicativeWords = analyzer.getMostIndicativeWords(10);
        
        System.out.println("Top positive words:");
        List<String> positiveWords = indicativeWords.get("positive");
        for (int i = 0; i < positiveWords.size(); i++) {
            System.out.printf("  %d. %s%n", i + 1, positiveWords.get(i));
        }
        System.out.println();
        
        System.out.println("Top negative words:");
        List<String> negativeWords = indicativeWords.get("negative");
        for (int i = 0; i < negativeWords.size(); i++) {
            System.out.printf("  %d. %s%n", i + 1, negativeWords.get(i));
        }
        System.out.println();
        
        // Evaluate on test data
        System.out.println("=== Model Evaluation ===");
        List<String> evaluationTexts = Arrays.asList(
            "I love this amazing product!",
            "This is terrible and awful.",
            "Great service and quality.",
            "Poor performance and bad quality.",
            "Excellent experience overall.",
            "Disappointing and frustrating.",
            "Wonderful and fantastic product.",
            "Horrible customer service.",
            "Outstanding value for money.",
            "Terrible waste of money."
        );
        
        List<Double> evaluationLabels = Arrays.asList(
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
        );
        
        Map<String, Double> metrics = analyzer.evaluate(evaluationTexts, evaluationLabels);
        
        System.out.println("Evaluation Metrics:");
        System.out.printf("  Accuracy:  %.4f%n", metrics.get("accuracy"));
        System.out.printf("  Precision: %.4f%n", metrics.get("precision"));
        System.out.printf("  Recall:    %.4f%n", metrics.get("recall"));
        System.out.printf("  F1 Score:  %.4f%n", metrics.get("f1_score"));
        System.out.println();
        
        // Interactive sentiment analysis
        System.out.println("=== Interactive Sentiment Analysis ===");
        List<String> interactiveTexts = Arrays.asList(
            "The movie was absolutely fantastic and I loved every minute of it!",
            "This restaurant has the worst food I've ever tasted.",
            "The book was okay, not great but not terrible either.",
            "Amazing performance by the actors, highly recommend this film!",
            "Terrible customer service, they were rude and unhelpful."
        );
        
        for (String text : interactiveTexts) {
            double sentiment = analyzer.predictSentiment(text);
            String sentimentClass = analyzer.predictSentimentClass(text);
            double confidence = analyzer.getSentimentConfidence(text);
            
            System.out.println("Text: " + text);
            System.out.printf("Sentiment: %.4f (%s) - Confidence: %.4f%n", 
                sentiment, sentimentClass, confidence);
            
            // Provide interpretation
            if (sentiment > 0.7) {
                System.out.println("  → Strongly positive sentiment");
            } else if (sentiment > 0.6) {
                System.out.println("  → Positive sentiment");
            } else if (sentiment > 0.4) {
                System.out.println("  → Neutral sentiment");
            } else if (sentiment > 0.3) {
                System.out.println("  → Negative sentiment");
            } else {
                System.out.println("  → Strongly negative sentiment");
            }
            System.out.println();
        }
        
        // Sentiment analysis with confidence thresholds
        System.out.println("=== Confidence-Based Classification ===");
        String uncertainText = "The product is okay, nothing special.";
        double sentiment = analyzer.predictSentiment(uncertainText);
        double confidence = analyzer.getSentimentConfidence(uncertainText);
        
        System.out.println("Text: " + uncertainText);
        System.out.printf("Sentiment: %.4f, Confidence: %.4f%n", sentiment, confidence);
        
        if (confidence < 0.3) {
            System.out.println("  → Low confidence: Consider as neutral/uncertain");
        } else {
            System.out.println("  → High confidence: Reliable prediction");
        }
        System.out.println();
        
        System.out.println("=== Sentiment Analysis Example Complete ===");
    }
}

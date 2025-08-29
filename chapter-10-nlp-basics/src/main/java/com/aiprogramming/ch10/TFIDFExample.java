package com.aiprogramming.ch10;

import java.util.Arrays;
import java.util.List;

/**
 * Example demonstrating TF-IDF functionality
 */
public class TFIDFExample {
    
    public static void main(String[] args) {
        System.out.println("=== TF-IDF Example ===\n");
        
        TFIDF tfidf = new TFIDF();
        
        // Sample documents
        List<String> documents = Arrays.asList(
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing helps computers understand and process human language.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "Text preprocessing is essential for natural language processing tasks.",
            "Artificial intelligence encompasses machine learning and deep learning techniques."
        );
        
        System.out.println("Documents:");
        for (int i = 0; i < documents.size(); i++) {
            System.out.println((i + 1) + ". " + documents.get(i));
        }
        System.out.println();
        
        // Fit and transform documents
        System.out.println("Training TF-IDF model...");
        List<double[]> tfidfVectors = tfidf.fitTransform(documents, true, false);
        
        // Print TF-IDF statistics
        tfidf.printTFIDFStats();
        System.out.println();
        
        // Print vocabulary
        System.out.println("Vocabulary:");
        List<String> vocabulary = tfidf.getVocabulary();
        for (int i = 0; i < Math.min(20, vocabulary.size()); i++) {
            System.out.printf("%2d. %s (IDF: %.4f)%n", i + 1, vocabulary.get(i), tfidf.getIDFScore(vocabulary.get(i)));
        }
        if (vocabulary.size() > 20) {
            System.out.println("... and " + (vocabulary.size() - 20) + " more words");
        }
        System.out.println();
        
        // Show TF-IDF vectors for first few documents
        System.out.println("TF-IDF Vectors (first 10 dimensions):");
        for (int i = 0; i < Math.min(3, tfidfVectors.size()); i++) {
            System.out.printf("Document %d: [", i + 1);
            double[] vector = tfidfVectors.get(i);
            for (int j = 0; j < Math.min(10, vector.length); j++) {
                System.out.printf("%.4f", vector[j]);
                if (j < Math.min(9, vector.length - 1)) {
                    System.out.print(", ");
                }
            }
            if (vector.length > 10) {
                System.out.print(", ...");
            }
            System.out.println("]");
        }
        System.out.println();
        
        // Document similarity example
        System.out.println("=== Document Similarity Example ===");
        String query = "machine learning algorithms";
        System.out.println("Query: " + query);
        
        // Transform query to TF-IDF vector
        double[] queryVector = tfidf.transformDocument(query, true, false);
        
        // Find most similar documents
        List<Integer> similarDocs = tfidf.findMostSimilarDocuments(queryVector, tfidfVectors, 3);
        
        System.out.println("Most similar documents:");
        for (int i = 0; i < similarDocs.size(); i++) {
            int docIndex = similarDocs.get(i);
            double similarity = tfidf.cosineSimilarity(queryVector, tfidfVectors.get(docIndex));
            System.out.printf("%d. Document %d (similarity: %.4f): %s%n", 
                i + 1, docIndex + 1, similarity, documents.get(docIndex));
        }
        System.out.println();
        
        // Compare all document pairs
        System.out.println("=== Document Similarity Matrix ===");
        System.out.print("Doc\\Doc");
        for (int i = 1; i <= documents.size(); i++) {
            System.out.printf("%8d", i);
        }
        System.out.println();
        
        for (int i = 0; i < documents.size(); i++) {
            System.out.printf("%6d", i + 1);
            for (int j = 0; j < documents.size(); j++) {
                double similarity = tfidf.cosineSimilarity(tfidfVectors.get(i), tfidfVectors.get(j));
                System.out.printf("%8.3f", similarity);
            }
            System.out.println();
        }
        System.out.println();
        
        // Top words by IDF (most distinctive words)
        System.out.println("=== Top Words by IDF (Most Distinctive) ===");
        List<String> topWords = tfidf.getTopWordsByIDF(10);
        for (int i = 0; i < topWords.size(); i++) {
            String word = topWords.get(i);
            System.out.printf("%d. %s (IDF: %.4f)%n", i + 1, word, tfidf.getIDFScore(word));
        }
        System.out.println();
        
        // Feature importance example
        System.out.println("=== Feature Importance Example ===");
        String testDoc = "Machine learning algorithms are used in artificial intelligence applications.";
        System.out.println("Test document: " + testDoc);
        
        double[] testVector = tfidf.transformDocument(testDoc, true, false);
        
        // Find words with highest TF-IDF scores
        System.out.println("Top words by TF-IDF score:");
        List<String> docTokens = new TextPreprocessor().preprocess(testDoc, true, false);
        for (String token : docTokens) {
            // Find the word in vocabulary and get its index
            List<String> vocab = tfidf.getVocabulary();
            for (int i = 0; i < vocab.size(); i++) {
                if (vocab.get(i).equals(token)) {
                    double score = testVector[i];
                    if (score > 0) {
                        System.out.printf("  %s: %.4f%n", token, score);
                    }
                    break;
                }
            }
        }
        System.out.println();
        
        // Bag of Words comparison
        System.out.println("=== Bag of Words vs TF-IDF Comparison ===");
        BagOfWords bow = new BagOfWords();
        List<double[]> bowVectors = bow.fitTransform(documents, true, false);
        
        System.out.println("Document 1 - Bag of Words vs TF-IDF (first 10 dimensions):");
        double[] bowVector = bowVectors.get(0);
        double[] tfidfVector = tfidfVectors.get(0);
        
        System.out.print("BoW:  [");
        for (int i = 0; i < Math.min(10, bowVector.length); i++) {
            System.out.printf("%.0f", bowVector[i]);
            if (i < Math.min(9, bowVector.length - 1)) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        
        System.out.print("TF-IDF: [");
        for (int i = 0; i < Math.min(10, tfidfVector.length); i++) {
            System.out.printf("%.4f", tfidfVector[i]);
            if (i < Math.min(9, tfidfVector.length - 1)) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
        System.out.println();
        
        System.out.println("=== TF-IDF Example Complete ===");
    }
}

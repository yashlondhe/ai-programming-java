# Chapter 14: Recommender Systems

## Introduction

Recommender systems are algorithms designed to suggest relevant items to users based on their preferences, behavior, and the characteristics of items. These systems have become ubiquitous in modern applications, from e-commerce platforms suggesting products to streaming services recommending movies and music. They help users discover new content while increasing engagement and satisfaction.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts and types of recommender systems
- Implement collaborative filtering algorithms (user-based and item-based)
- Build content-based filtering systems using item features
- Create hybrid recommender systems that combine multiple approaches
- Evaluate recommender system performance using appropriate metrics
- Apply matrix factorization techniques for dimensionality reduction
- Design scalable recommendation architectures for real-world applications

### Key Concepts

- **Collaborative Filtering**: Recommending items based on similar users' preferences
- **Content-Based Filtering**: Recommending items similar to what the user has liked before
- **Hybrid Systems**: Combining multiple recommendation approaches for better results
- **Matrix Factorization**: Dimensionality reduction techniques for large datasets
- **Evaluation Metrics**: Precision, recall, MAE, RMSE for measuring recommendation quality
- **Cold Start Problem**: Challenges with new users or items having no history

## 14.1 Types of Recommender Systems

Recommender systems can be broadly classified into several categories based on their approach to generating recommendations.

### 14.1.1 Collaborative Filtering

Collaborative filtering (CF) is one of the most popular approaches, which makes recommendations based on the preferences of similar users or similar items.

#### User-Based Collaborative Filtering

User-based CF finds users with similar preferences and recommends items they liked.

```java
package com.aiprogramming.ch14.collaborative;

import com.aiprogramming.ch14.core.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * User-based collaborative filtering recommender system.
 * Finds similar users and recommends items they liked.
 */
public class UserBasedCollaborativeFiltering implements RecommenderSystem {
    
    private final List<Rating> ratings;
    private final Map<Integer, List<Rating>> userRatings;
    private final Map<Integer, Double> userAverages;
    private final Map<String, Double> similarityCache;
    private final int minCommonItems;
    private final SimilarityMetric similarityMetric;
    
    /**
     * Similarity metrics for comparing users.
     */
    public enum SimilarityMetric {
        PEARSON, COSINE, EUCLIDEAN
    }
    
    /**
     * Constructor with default settings.
     * 
     * @param ratings list of all ratings
     */
    public UserBasedCollaborativeFiltering(List<Rating> ratings) {
        this(ratings, SimilarityMetric.PEARSON, 3);
    }
    
    /**
     * Constructor with custom settings.
     * 
     * @param ratings list of all ratings
     * @param similarityMetric the similarity metric to use
     * @param minCommonItems minimum number of common items for similarity calculation
     */
    public UserBasedCollaborativeFiltering(List<Rating> ratings, SimilarityMetric similarityMetric, int minCommonItems) {
        this.ratings = new ArrayList<>(ratings);
        this.similarityMetric = similarityMetric;
        this.minCommonItems = minCommonItems;
        this.similarityCache = new HashMap<>();
        
        // Build data structures
        this.userRatings = buildUserRatings();
        this.userAverages = calculateUserAverages();
    }
    
    @Override
    public List<RecommendationResult> recommend(int userId, int numRecommendations) {
        if (!userRatings.containsKey(userId)) {
            return new ArrayList<>();
        }
        
        // Find similar users
        List<UserSimilarity> similarUsers = findSimilarUsers(userId, 20);
        
        // Get items rated by similar users but not by target user
        Set<Integer> targetUserItems = userRatings.get(userId).stream()
                .map(Rating::getItemId)
                .collect(Collectors.toSet());
        
        Map<Integer, Double> itemScores = new HashMap<>();
        Map<Integer, Integer> itemCounts = new HashMap<>();
        
        for (UserSimilarity userSim : similarUsers) {
            int similarUserId = userSim.userId;
            double similarity = userSim.similarity;
            
            for (Rating rating : userRatings.get(similarUserId)) {
                int itemId = rating.getItemId();
                
                // Skip items already rated by target user
                if (targetUserItems.contains(itemId)) {
                    continue;
                }
                
                // Calculate weighted score
                double userAvg = userAverages.get(similarUserId);
                double adjustedRating = rating.getRating() - userAvg;
                double weightedScore = similarity * adjustedRating;
                
                itemScores.merge(itemId, weightedScore, Double::sum);
                itemCounts.merge(itemId, 1, Integer::sum);
            }
        }
        
        // Calculate final scores and create recommendations
        List<RecommendationResult> recommendations = new ArrayList<>();
        for (Map.Entry<Integer, Double> entry : itemScores.entrySet()) {
            int itemId = entry.getKey();
            double totalScore = entry.getValue();
            int count = itemCounts.get(itemId);
            
            if (count >= 2) { // At least 2 similar users rated this item
                double avgScore = totalScore / count;
                double userAvg = userAverages.get(userId);
                double predictedRating = userAvg + avgScore;
                
                // Clamp to valid rating range
                predictedRating = Math.max(1.0, Math.min(5.0, predictedRating));
                
                recommendations.add(new RecommendationResult(itemId, predictedRating));
            }
        }
        
        // Sort by score and return top N
        recommendations.sort(Collections.reverseOrder());
        return recommendations.stream()
                .limit(numRecommendations)
                .collect(Collectors.toList());
    }
    
    /**
     * Calculate similarity between two users using Pearson correlation.
     */
    private double calculatePearsonCorrelation(List<Double> vector1, List<Double> vector2) {
        int n = vector1.size();
        if (n == 0) return 0.0;
        
        double sum1 = vector1.stream().mapToDouble(Double::doubleValue).sum();
        double sum2 = vector2.stream().mapToDouble(Double::doubleValue).sum();
        double sum1Sq = vector1.stream().mapToDouble(x -> x * x).sum();
        double sum2Sq = vector2.stream().mapToDouble(x -> x * x).sum();
        double pSum = 0.0;
        
        for (int i = 0; i < n; i++) {
            pSum += vector1.get(i) * vector2.get(i);
        }
        
        double num = pSum - (sum1 * sum2 / n);
        double den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
        
        return den == 0 ? 0 : num / den;
    }
}
```

#### Item-Based Collaborative Filtering

Item-based CF finds similar items and recommends them to users who liked related items.

### 14.1.2 Content-Based Filtering

Content-based filtering recommends items similar to what the user has liked before, based on item features.

#### Feature Vectors

```java
package com.aiprogramming.ch14.content;

import java.util.*;

/**
 * Represents a feature vector for content-based filtering.
 * Can handle both categorical and numerical features.
 */
public class FeatureVector {
    private final Map<String, Double> features;
    private final Map<String, String> categoricalFeatures;
    private final String itemId;
    
    /**
     * Constructor for simple string-based features (e.g., movie genres).
     * 
     * @param itemId the ID of the item
     * @param featureStrings array of feature strings
     */
    public FeatureVector(String itemId, String... featureStrings) {
        this.itemId = itemId;
        this.features = new HashMap<>();
        this.categoricalFeatures = new HashMap<>();
        
        for (String feature : featureStrings) {
            if (feature != null && !feature.trim().isEmpty()) {
                this.categoricalFeatures.put(feature.trim().toLowerCase(), "1");
            }
        }
    }
    
    /**
     * Calculate cosine similarity with another feature vector.
     * 
     * @param other the other feature vector
     * @return cosine similarity score between 0 and 1
     */
    public double cosineSimilarity(FeatureVector other) {
        if (other == null) return 0.0;
        
        // Calculate dot product of numerical features
        double dotProduct = 0.0;
        Set<String> allFeatures = new HashSet<>(features.keySet());
        allFeatures.addAll(other.features.keySet());
        
        for (String feature : allFeatures) {
            dotProduct += getFeature(feature) * other.getFeature(feature);
        }
        
        // Calculate magnitudes
        double magnitude1 = getMagnitude();
        double magnitude2 = other.getMagnitude();
        
        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (magnitude1 * magnitude2);
    }
    
    /**
     * Calculate Jaccard similarity for categorical features.
     * 
     * @param other the other feature vector
     * @return Jaccard similarity score between 0 and 1
     */
    public double jaccardSimilarity(FeatureVector other) {
        if (other == null) return 0.0;
        
        Set<String> features1 = new HashSet<>(categoricalFeatures.keySet());
        Set<String> features2 = new HashSet<>(other.categoricalFeatures.keySet());
        
        if (features1.isEmpty() && features2.isEmpty()) {
            return 1.0;
        }
        
        Set<String> intersection = new HashSet<>(features1);
        intersection.retainAll(features2);
        
        Set<String> union = new HashSet<>(features1);
        union.addAll(features2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    /**
     * Calculate overall similarity combining numerical and categorical features.
     * 
     * @param other the other feature vector
     * @param numericalWeight weight for numerical similarity (0-1)
     * @return combined similarity score
     */
    public double combinedSimilarity(FeatureVector other, double numericalWeight) {
        double categoricalWeight = 1.0 - numericalWeight;
        
        double numericalSim = cosineSimilarity(other);
        double categoricalSim = jaccardSimilarity(other);
        
        return numericalWeight * numericalSim + categoricalWeight * categoricalSim;
    }
}
```

#### Content-Based Recommender

```java
package com.aiprogramming.ch14.content;

import com.aiprogramming.ch14.core.RecommendationResult;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Content-based filtering recommender system.
 * Recommends items similar to what the user has liked before based on item features.
 */
public class ContentBasedFiltering {
    
    private final Map<String, FeatureVector> items;
    private final Map<String, Double> itemAverages;
    
    /**
     * Constructor.
     */
    public ContentBasedFiltering() {
        this.items = new HashMap<>();
        this.itemAverages = new HashMap<>();
    }
    
    /**
     * Add an item with its features to the recommender.
     * 
     * @param itemFeatures the item's feature vector
     */
    public void addItem(FeatureVector itemFeatures) {
        items.put(itemFeatures.getItemId(), itemFeatures);
    }
    
    /**
     * Generate recommendations based on user profile.
     * 
     * @param userProfile the user's feature preferences
     * @param numRecommendations number of recommendations to generate
     * @return list of recommendations sorted by relevance
     */
    public List<RecommendationResult> recommend(FeatureVector userProfile, int numRecommendations) {
        List<RecommendationResult> recommendations = new ArrayList<>();
        
        for (FeatureVector itemFeatures : items.values()) {
            // Calculate similarity between user profile and item
            double similarity = userProfile.combinedSimilarity(itemFeatures, 0.5);
            
            if (similarity > 0) {
                String itemId = itemFeatures.getItemId();
                double score = similarity;
                
                // Boost score if we have rating information
                if (itemAverages.containsKey(itemId)) {
                    double avgRating = itemAverages.get(itemId);
                    score = similarity * (avgRating / 5.0); // Normalize rating to 0-1
                }
                
                recommendations.add(new RecommendationResult(itemId, score));
            }
        }
        
        // Sort by score and return top N
        recommendations.sort(Collections.reverseOrder());
        return recommendations.stream()
                .limit(numRecommendations)
                .collect(Collectors.toList());
    }
}
```

### 14.1.3 Hybrid Systems

Hybrid recommender systems combine multiple approaches to overcome the limitations of individual methods.

```java
package com.aiprogramming.ch14.hybrid;

import com.aiprogramming.ch14.collaborative.UserBasedCollaborativeFiltering;
import com.aiprogramming.ch14.content.ContentBasedFiltering;
import com.aiprogramming.ch14.content.FeatureVector;
import com.aiprogramming.ch14.core.RecommendationResult;
import java.util.*;

/**
 * Hybrid recommender system that combines multiple recommendation approaches.
 * Supports weighted combinations of collaborative filtering and content-based filtering.
 */
public class HybridRecommender {
    
    private final List<RecommenderComponent> recommenders;
    
    /**
     * Constructor.
     */
    public HybridRecommender() {
        this.recommenders = new ArrayList<>();
    }
    
    /**
     * Add a collaborative filtering recommender with weight.
     * 
     * @param collaborativeRecommender the collaborative filtering recommender
     * @param weight the weight for this recommender (0-1)
     */
    public void addRecommender(UserBasedCollaborativeFiltering collaborativeRecommender, double weight) {
        recommenders.add(new RecommenderComponent(collaborativeRecommender, weight, RecommenderType.COLLABORATIVE));
    }
    
    /**
     * Add a content-based filtering recommender with weight.
     * 
     * @param contentRecommender the content-based filtering recommender
     * @param weight the weight for this recommender (0-1)
     */
    public void addRecommender(ContentBasedFiltering contentRecommender, double weight) {
        recommenders.add(new RecommenderComponent(contentRecommender, weight, RecommenderType.CONTENT));
    }
    
    /**
     * Generate hybrid recommendations.
     * 
     * @param userId the user ID for collaborative filtering
     * @param userProfile the user profile for content-based filtering
     * @param numRecommendations number of recommendations to generate
     * @return combined recommendations
     */
    public List<RecommendationResult> recommend(int userId, FeatureVector userProfile, int numRecommendations) {
        Map<String, Double> combinedScores = new HashMap<>();
        Map<String, Integer> itemCounts = new HashMap<>();
        
        // Get recommendations from each component
        for (RecommenderComponent component : recommenders) {
            List<RecommendationResult> componentRecommendations = new ArrayList<>();
            
            switch (component.type) {
                case COLLABORATIVE:
                    UserBasedCollaborativeFiltering collaborativeRecommender = 
                        (UserBasedCollaborativeFiltering) component.recommender;
                    componentRecommendations = collaborativeRecommender.recommend(userId, numRecommendations * 2);
                    break;
                    
                case CONTENT:
                    ContentBasedFiltering contentRecommender = 
                        (ContentBasedFiltering) component.recommender;
                    componentRecommendations = contentRecommender.recommend(userProfile, numRecommendations * 2);
                    break;
            }
            
            // Combine scores with weights
            for (RecommendationResult rec : componentRecommendations) {
                String itemId = rec.getItemId().toString();
                double weightedScore = rec.getScore() * component.weight;
                
                combinedScores.merge(itemId, weightedScore, Double::sum);
                itemCounts.merge(itemId, 1, Integer::sum);
            }
        }
        
        // Create final recommendations
        List<RecommendationResult> finalRecommendations = new ArrayList<>();
        for (Map.Entry<String, Double> entry : combinedScores.entrySet()) {
            String itemId = entry.getKey();
            double totalScore = entry.getValue();
            int count = itemCounts.get(itemId);
            
            // Average the scores from different recommenders
            double avgScore = totalScore / count;
            finalRecommendations.add(new RecommendationResult(itemId, avgScore));
        }
        
        // Sort by score and return top N
        finalRecommendations.sort(Collections.reverseOrder());
        return finalRecommendations.stream()
                .limit(numRecommendations)
                .collect(Collectors.toList());
    }
}
```

## 14.2 Core Data Structures

### 14.2.1 Rating System

```java
package com.aiprogramming.ch14.core;

import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Represents a user's rating for an item.
 * This is the core data structure used by all recommender systems.
 */
public class Rating {
    private final int userId;
    private final int itemId;
    private final double rating;
    private final LocalDateTime timestamp;
    
    /**
     * Constructor for a rating without timestamp.
     * 
     * @param userId the ID of the user
     * @param itemId the ID of the item
     * @param rating the rating value (typically 1-5 scale)
     */
    public Rating(int userId, int itemId, double rating) {
        this(userId, itemId, rating, LocalDateTime.now());
    }
    
    /**
     * Constructor for a rating with timestamp.
     * 
     * @param userId the ID of the user
     * @param itemId the ID of the item
     * @param rating the rating value (typically 1-5 scale)
     * @param timestamp when the rating was given
     */
    public Rating(int userId, int itemId, double rating, LocalDateTime timestamp) {
        this.userId = userId;
        this.itemId = itemId;
        this.rating = rating;
        this.timestamp = timestamp;
        
        // Validate rating value
        if (rating < 0 || rating > 5) {
            throw new IllegalArgumentException("Rating must be between 0 and 5");
        }
    }
    
    /**
     * Check if this rating is positive (above a threshold).
     * 
     * @param threshold the threshold to compare against (default 3.5)
     * @return true if rating is above threshold
     */
    public boolean isPositive(double threshold) {
        return rating >= threshold;
    }
    
    /**
     * Get the normalized rating (0-1 scale).
     * 
     * @return normalized rating
     */
    public double getNormalizedRating() {
        return rating / 5.0;
    }
}
```

### 14.2.2 Recommendation Results

```java
package com.aiprogramming.ch14.core;

import java.util.Objects;

/**
 * Represents a single recommendation result with item ID and relevance score.
 * Used to return recommendations from recommender systems.
 */
public class RecommendationResult implements Comparable<RecommendationResult> {
    private final Object itemId;
    private final double score;
    private final String reason;
    
    /**
     * Constructor for a recommendation result.
     * 
     * @param itemId the ID of the recommended item
     * @param score the relevance score (higher is better)
     */
    public RecommendationResult(Object itemId, double score) {
        this(itemId, score, "");
    }
    
    /**
     * Constructor for a recommendation result with explanation.
     * 
     * @param itemId the ID of the recommended item
     * @param score the relevance score (higher is better)
     * @param reason explanation for why this item was recommended
     */
    public RecommendationResult(Object itemId, double score, String reason) {
        this.itemId = itemId;
        this.score = score;
        this.reason = reason != null ? reason : "";
    }
    
    /**
     * Compare this recommendation with another based on score.
     * Higher scores come first (descending order).
     * 
     * @param other the other recommendation to compare with
     * @return negative if this score is higher, positive if other is higher, 0 if equal
     */
    @Override
    public int compareTo(RecommendationResult other) {
        return Double.compare(other.score, this.score); // Descending order
    }
}
```

## 14.3 Evaluation Metrics

### 14.3.1 Precision and Recall

```java
package com.aiprogramming.ch14.evaluation;

import com.aiprogramming.ch14.core.RecommendationResult;
import com.aiprogramming.ch14.core.Rating;
import java.util.*;

/**
 * Calculates precision and recall metrics for recommender systems.
 */
public class PrecisionRecall {
    
    /**
     * Calculate precision@k for recommendations.
     * 
     * @param recommendations the recommended items
     * @param relevantItems set of relevant items for the user
     * @param k the number of top recommendations to consider
     * @return precision@k score
     */
    public static double calculatePrecisionAtK(List<RecommendationResult> recommendations, 
                                            Set<Object> relevantItems, int k) {
        if (k <= 0 || recommendations.isEmpty()) {
            return 0.0;
        }
        
        int relevantCount = 0;
        int topK = Math.min(k, recommendations.size());
        
        for (int i = 0; i < topK; i++) {
            Object itemId = recommendations.get(i).getItemId();
            if (relevantItems.contains(itemId)) {
                relevantCount++;
            }
        }
        
        return (double) relevantCount / topK;
    }
    
    /**
     * Calculate recall@k for recommendations.
     * 
     * @param recommendations the recommended items
     * @param relevantItems set of relevant items for the user
     * @param k the number of top recommendations to consider
     * @return recall@k score
     */
    public static double calculateRecallAtK(List<RecommendationResult> recommendations, 
                                         Set<Object> relevantItems, int k) {
        if (relevantItems.isEmpty()) {
            return 0.0;
        }
        
        int relevantCount = 0;
        int topK = Math.min(k, recommendations.size());
        
        for (int i = 0; i < topK; i++) {
            Object itemId = recommendations.get(i).getItemId();
            if (relevantItems.contains(itemId)) {
                relevantCount++;
            }
        }
        
        return (double) relevantCount / relevantItems.size();
    }
    
    /**
     * Calculate F1 score (harmonic mean of precision and recall).
     * 
     * @param precision the precision score
     * @param recall the recall score
     * @return F1 score
     */
    public static double calculateF1Score(double precision, double recall) {
        if (precision + recall == 0) {
            return 0.0;
        }
        return 2 * (precision * recall) / (precision + recall);
    }
}
```

### 14.3.2 Mean Absolute Error

```java
package com.aiprogramming.ch14.evaluation;

import com.aiprogramming.ch14.core.Rating;
import java.util.*;

/**
 * Calculates Mean Absolute Error (MAE) for rating predictions.
 */
public class MeanAbsoluteError {
    
    /**
     * Calculate MAE for predicted ratings.
     * 
     * @param predictions map of (userId, itemId) to predicted rating
     * @param actualRatings list of actual ratings
     * @return MAE score
     */
    public static double calculateMAE(Map<String, Double> predictions, List<Rating> actualRatings) {
        if (predictions.isEmpty() || actualRatings.isEmpty()) {
            return 0.0;
        }
        
        double totalError = 0.0;
        int count = 0;
        
        for (Rating rating : actualRatings) {
            String key = rating.getUserId() + "_" + rating.getItemId();
            Double predictedRating = predictions.get(key);
            
            if (predictedRating != null) {
                totalError += Math.abs(predictedRating - rating.getRating());
                count++;
            }
        }
        
        return count > 0 ? totalError / count : 0.0;
    }
    
    /**
     * Calculate Root Mean Square Error (RMSE).
     * 
     * @param predictions map of (userId, itemId) to predicted rating
     * @param actualRatings list of actual ratings
     * @return RMSE score
     */
    public static double calculateRMSE(Map<String, Double> predictions, List<Rating> actualRatings) {
        if (predictions.isEmpty() || actualRatings.isEmpty()) {
            return 0.0;
        }
        
        double totalSquaredError = 0.0;
        int count = 0;
        
        for (Rating rating : actualRatings) {
            String key = rating.getUserId() + "_" + rating.getItemId();
            Double predictedRating = predictions.get(key);
            
            if (predictedRating != null) {
                double error = predictedRating - rating.getRating();
                totalSquaredError += error * error;
                count++;
            }
        }
        
        return count > 0 ? Math.sqrt(totalSquaredError / count) : 0.0;
    }
}
```

## 14.4 Practical Example: Movie Recommendation System

### 14.4.1 Complete Implementation

```java
package com.aiprogramming.ch14;

import com.aiprogramming.ch14.collaborative.UserBasedCollaborativeFiltering;
import com.aiprogramming.ch14.content.ContentBasedFiltering;
import com.aiprogramming.ch14.content.FeatureVector;
import com.aiprogramming.ch14.core.Rating;
import com.aiprogramming.ch14.core.RecommendationResult;
import com.aiprogramming.ch14.hybrid.HybridRecommender;

import java.util.*;

/**
 * Main example demonstrating recommender systems.
 * Shows collaborative filtering, content-based filtering, and hybrid approaches.
 */
public class MovieRecommenderExample {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 14: Recommender Systems Example ===\n");
        
        // Create sample movie data
        List<Rating> ratings = createSampleRatings();
        Map<String, FeatureVector> movieFeatures = createMovieFeatures();
        
        System.out.println("Sample data created:");
        System.out.println("- " + ratings.size() + " ratings from " + 
                          ratings.stream().mapToInt(Rating::getUserId).distinct().count() + " users");
        System.out.println("- " + movieFeatures.size() + " movies with features\n");
        
        // 1. User-based Collaborative Filtering
        System.out.println("1. User-based Collaborative Filtering");
        System.out.println("=====================================");
        demonstrateCollaborativeFiltering(ratings);
        
        // 2. Content-based Filtering
        System.out.println("\n2. Content-based Filtering");
        System.out.println("===========================");
        demonstrateContentBasedFiltering(movieFeatures);
        
        // 3. Hybrid Recommender
        System.out.println("\n3. Hybrid Recommender System");
        System.out.println("=============================");
        demonstrateHybridRecommender(ratings, movieFeatures);
        
        // 4. Performance Comparison
        System.out.println("\n4. Performance Comparison");
        System.out.println("========================");
        comparePerformance(ratings, movieFeatures);
    }
    
    /**
     * Demonstrate user-based collaborative filtering.
     */
    private static void demonstrateCollaborativeFiltering(List<Rating> ratings) {
        UserBasedCollaborativeFiltering recommender = new UserBasedCollaborativeFiltering(ratings);
        
        // Get recommendations for user 1
        int targetUser = 1;
        List<RecommendationResult> recommendations = recommender.recommend(targetUser, 5);
        
        System.out.println("Recommendations for User " + targetUser + ":");
        for (int i = 0; i < recommendations.size(); i++) {
            RecommendationResult rec = recommendations.get(i);
            System.out.printf("  %d. Movie %s (Score: %.3f)\n", 
                            i + 1, rec.getItemId(), rec.getScore());
        }
        
        // Show user similarities
        System.out.println("\nUser similarities with User " + targetUser + ":");
        for (int userId = 2; userId <= 5; userId++) {
            double similarity = recommender.getUserSimilarity(targetUser, userId);
            System.out.printf("  User %d: %.3f\n", userId, similarity);
        }
        
        // Predict rating for a specific movie
        int movieId = 3;
        double predictedRating = recommender.predictRating(targetUser, movieId);
        System.out.printf("\nPredicted rating for User %d on Movie %d: %.2f\n", 
                         targetUser, movieId, predictedRating);
    }
    
    /**
     * Demonstrate content-based filtering.
     */
    private static void demonstrateContentBasedFiltering(Map<String, FeatureVector> movieFeatures) {
        ContentBasedFiltering recommender = new ContentBasedFiltering();
        
        // Add movie features to the recommender
        for (FeatureVector features : movieFeatures.values()) {
            recommender.addItem(features);
        }
        
        // Create user profile based on liked movies
        FeatureVector userProfile = new FeatureVector("user1", 
            "action", "adventure", "sci-fi"); // User likes action/adventure/sci-fi movies
        
        System.out.println("User profile: " + userProfile.getAllFeatureNames());
        
        // Get recommendations
        List<RecommendationResult> recommendations = recommender.recommend(userProfile, 5);
        
        System.out.println("Content-based recommendations:");
        for (int i = 0; i < recommendations.size(); i++) {
            RecommendationResult rec = recommendations.get(i);
            System.out.printf("  %d. Movie %s (Score: %.3f)\n", 
                            i + 1, rec.getItemId(), rec.getScore());
        }
        
        // Show item similarities
        System.out.println("\nMovie similarities:");
        String[] movieIds = {"1", "2", "3", "4", "5"};
        for (int i = 0; i < movieIds.length; i++) {
            for (int j = i + 1; j < movieIds.length; j++) {
                FeatureVector movie1 = movieFeatures.get(movieIds[i]);
                FeatureVector movie2 = movieFeatures.get(movieIds[j]);
                if (movie1 != null && movie2 != null) {
                    double similarity = movie1.combinedSimilarity(movie2, 0.5);
                    System.out.printf("  Movie %s vs Movie %s: %.3f\n", 
                                    movieIds[i], movieIds[j], similarity);
                }
            }
        }
    }
    
    /**
     * Demonstrate hybrid recommender system.
     */
    private static void demonstrateHybridRecommender(List<Rating> ratings, 
                                                   Map<String, FeatureVector> movieFeatures) {
        // Create individual recommenders
        UserBasedCollaborativeFiltering collaborativeRecommender = 
            new UserBasedCollaborativeFiltering(ratings);
        
        ContentBasedFiltering contentRecommender = new ContentBasedFiltering();
        for (FeatureVector features : movieFeatures.values()) {
            contentRecommender.addItem(features);
        }
        
        // Create hybrid recommender
        HybridRecommender hybridRecommender = new HybridRecommender();
        hybridRecommender.addRecommender(collaborativeRecommender, 0.6); // 60% weight
        hybridRecommender.addRecommender(contentRecommender, 0.4);       // 40% weight
        
        // Get hybrid recommendations
        int targetUser = 1;
        FeatureVector userProfile = new FeatureVector("user1", "action", "adventure");
        
        List<RecommendationResult> hybridRecommendations = 
            hybridRecommender.recommend(targetUser, userProfile, 5);
        
        System.out.println("Hybrid recommendations for User " + targetUser + ":");
        for (int i = 0; i < hybridRecommendations.size(); i++) {
            RecommendationResult rec = hybridRecommendations.get(i);
            System.out.printf("  %d. Movie %s (Score: %.3f)\n", 
                            i + 1, rec.getItemId(), rec.getScore());
        }
    }
    
    /**
     * Create sample rating data.
     */
    private static List<Rating> createSampleRatings() {
        List<Rating> ratings = new ArrayList<>();
        
        // User 1 ratings
        ratings.add(new Rating(1, 1, 5.0)); // Action movie
        ratings.add(new Rating(1, 2, 4.0)); // Adventure movie
        ratings.add(new Rating(1, 3, 3.0)); // Comedy movie
        ratings.add(new Rating(1, 4, 2.0)); // Drama movie
        
        // User 2 ratings
        ratings.add(new Rating(2, 1, 3.0));
        ratings.add(new Rating(2, 2, 4.0));
        ratings.add(new Rating(2, 3, 5.0)); // Likes comedy
        ratings.add(new Rating(2, 4, 4.0));
        ratings.add(new Rating(2, 5, 5.0)); // Romance movie
        
        // User 3 ratings
        ratings.add(new Rating(3, 1, 4.0));
        ratings.add(new Rating(3, 2, 5.0)); // Likes adventure
        ratings.add(new Rating(3, 3, 2.0));
        ratings.add(new Rating(3, 4, 5.0)); // Likes drama
        ratings.add(new Rating(3, 5, 3.0));
        
        // User 4 ratings
        ratings.add(new Rating(4, 1, 5.0)); // Likes action
        ratings.add(new Rating(4, 2, 4.0));
        ratings.add(new Rating(4, 3, 1.0));
        ratings.add(new Rating(4, 4, 3.0));
        ratings.add(new Rating(4, 5, 2.0));
        
        // User 5 ratings
        ratings.add(new Rating(5, 1, 2.0));
        ratings.add(new Rating(5, 2, 3.0));
        ratings.add(new Rating(5, 3, 5.0)); // Likes comedy
        ratings.add(new Rating(5, 4, 4.0));
        ratings.add(new Rating(5, 5, 5.0)); // Likes romance
        
        return ratings;
    }
    
    /**
     * Create sample movie features.
     */
    private static Map<String, FeatureVector> createMovieFeatures() {
        Map<String, FeatureVector> features = new HashMap<>();
        
        // Movie 1: Action/Adventure/Sci-fi
        features.put("1", new FeatureVector("1", "action", "adventure", "sci-fi"));
        
        // Movie 2: Adventure/Fantasy
        features.put("2", new FeatureVector("2", "adventure", "fantasy"));
        
        // Movie 3: Comedy/Romance
        features.put("3", new FeatureVector("3", "comedy", "romance"));
        
        // Movie 4: Drama/Thriller
        features.put("4", new FeatureVector("4", "drama", "thriller"));
        
        // Movie 5: Romance/Drama
        features.put("5", new FeatureVector("5", "romance", "drama"));
        
        return features;
    }
}
```

## 14.5 Advanced Topics

### 14.5.1 Matrix Factorization

Matrix factorization techniques like Singular Value Decomposition (SVD) can be used to learn latent factors from the user-item rating matrix.

### 14.5.2 Cold Start Problem

The cold start problem occurs when new users or items have no rating history, making it difficult to generate accurate recommendations.

### 14.5.3 Scalability Considerations

For large-scale applications, consider:
- Approximate nearest neighbor algorithms
- Distributed computing frameworks
- Caching strategies
- Incremental updates

## 14.6 Summary

In this chapter, we explored the fundamental concepts and implementations of recommender systems:

### Key Takeaways

1. **Collaborative Filtering**: Uses similarity between users or items to make recommendations
2. **Content-Based Filtering**: Recommends items similar to what the user has liked before
3. **Hybrid Systems**: Combine multiple approaches for better recommendation quality
4. **Evaluation Metrics**: Use appropriate metrics like precision, recall, MAE, and RMSE
5. **Scalability**: Consider performance implications for large datasets

### Practical Applications

- **E-commerce**: Product recommendations
- **Streaming Services**: Movie, music, and video recommendations
- **Social Media**: Content and connection suggestions
- **News Aggregation**: Article recommendations

### Next Steps

- Explore matrix factorization techniques
- Implement real-time recommendation systems
- Study deep learning approaches for recommendations
- Consider multi-objective optimization for recommendation quality

## Exercises

### Exercise 14.1: Implement Item-Based Collaborative Filtering

Create an item-based collaborative filtering system that finds similar items and recommends them to users.

### Exercise 14.2: Build a Matrix Factorization Recommender

Implement a matrix factorization approach using SVD to learn latent factors from the rating matrix.

### Exercise 14.3: Create a Multi-Criteria Recommender

Build a recommender system that considers multiple criteria (e.g., price, rating, popularity) when making recommendations.

### Exercise 14.4: Implement Real-Time Recommendations

Create a system that can update recommendations in real-time as users interact with items.

### Exercise 14.5: Build a Context-Aware Recommender

Implement a recommender that considers context (time, location, device) when making recommendations.

## Further Reading

1. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
2. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
3. Jannach, D., Zanker, M., Felfernig, A., & Friedrich, G. (2010). *Recommender Systems: An Introduction*. Cambridge University Press.
4. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.
5. Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. *Advances in artificial intelligence*, 2009.

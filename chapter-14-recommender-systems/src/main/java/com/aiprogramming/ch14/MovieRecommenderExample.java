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
            System.out.printf("  %d. Movie %d (Score: %.3f)\n", 
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
        
        // Show individual recommender results for comparison
        System.out.println("\nIndividual recommender results:");
        
        List<RecommendationResult> collaborativeRecs = 
            collaborativeRecommender.recommend(targetUser, 3);
        System.out.println("  Collaborative filtering (top 3):");
        for (RecommendationResult rec : collaborativeRecs) {
            System.out.printf("    Movie %d (Score: %.3f)\n", rec.getItemId(), rec.getScore());
        }
        
        List<RecommendationResult> contentRecs = 
            contentRecommender.recommend(userProfile, 3);
        System.out.println("  Content-based filtering (top 3):");
        for (RecommendationResult rec : contentRecs) {
            System.out.printf("    Movie %s (Score: %.3f)\n", rec.getItemId(), rec.getScore());
        }
    }
    
    /**
     * Compare performance of different recommender approaches.
     */
    private static void comparePerformance(List<Rating> ratings, 
                                         Map<String, FeatureVector> movieFeatures) {
        // Create recommenders
        UserBasedCollaborativeFiltering collaborativeRecommender = 
            new UserBasedCollaborativeFiltering(ratings);
        
        ContentBasedFiltering contentRecommender = new ContentBasedFiltering();
        for (FeatureVector features : movieFeatures.values()) {
            contentRecommender.addItem(features);
        }
        
        HybridRecommender hybridRecommender = new HybridRecommender();
        hybridRecommender.addRecommender(collaborativeRecommender, 0.6);
        hybridRecommender.addRecommender(contentRecommender, 0.4);
        
        // Test with different users
        int[] testUsers = {1, 2, 3};
        FeatureVector[] userProfiles = {
            new FeatureVector("user1", "action", "adventure"),
            new FeatureVector("user2", "comedy", "romance"),
            new FeatureVector("user3", "drama", "thriller")
        };
        
        System.out.println("Performance comparison:");
        System.out.println("User\tCollaborative\tContent-based\tHybrid");
        System.out.println("----\t-------------\t-------------\t------");
        
        for (int i = 0; i < testUsers.length; i++) {
            int userId = testUsers[i];
            FeatureVector profile = userProfiles[i];
            
            // Measure recommendation diversity (number of unique items)
            Set<Integer> collaborativeItems = new HashSet<>();
            Set<String> contentItems = new HashSet<>();
            Set<String> hybridItems = new HashSet<>();
            
            List<RecommendationResult> collaborativeRecs = 
                collaborativeRecommender.recommend(userId, 10);
            List<RecommendationResult> contentRecs = 
                contentRecommender.recommend(profile, 10);
            List<RecommendationResult> hybridRecs = 
                hybridRecommender.recommend(userId, profile, 10);
            
            collaborativeRecs.forEach(rec -> collaborativeItems.add((Integer) rec.getItemId()));
            contentRecs.forEach(rec -> contentItems.add(rec.getItemId().toString()));
            hybridRecs.forEach(rec -> hybridItems.add(rec.getItemId().toString()));
            
            System.out.printf("%d\t%d\t\t%d\t\t%d\n", 
                            userId, collaborativeItems.size(), 
                            contentItems.size(), hybridItems.size());
        }
        
        // Show average recommendation scores
        System.out.println("\nAverage recommendation scores:");
        double collaborativeAvg = collaborativeRecommender.recommend(1, 10).stream()
                .mapToDouble(RecommendationResult::getScore)
                .average()
                .orElse(0.0);
        
        double contentAvg = contentRecommender.recommend(userProfiles[0], 10).stream()
                .mapToDouble(RecommendationResult::getScore)
                .average()
                .orElse(0.0);
        
        double hybridAvg = hybridRecommender.recommend(1, userProfiles[0], 10).stream()
                .mapToDouble(RecommendationResult::getScore)
                .average()
                .orElse(0.0);
        
        System.out.printf("Collaborative: %.3f\n", collaborativeAvg);
        System.out.printf("Content-based: %.3f\n", contentAvg);
        System.out.printf("Hybrid: %.3f\n", hybridAvg);
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

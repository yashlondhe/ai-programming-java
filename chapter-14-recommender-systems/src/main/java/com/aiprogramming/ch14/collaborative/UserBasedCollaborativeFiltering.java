package com.aiprogramming.ch14.collaborative;

import com.aiprogramming.ch14.core.*;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;

import java.util.*;
import java.util.stream.Collectors;

/**
 * User-based collaborative filtering recommender system.
 * Finds similar users and recommends items they liked.
 */
public class UserBasedCollaborativeFiltering implements RecommenderSystem {
    
    private final List<Rating> ratings;
    private final Map<Integer, List<Rating>> userRatings;
    private final Map<Integer, List<Rating>> itemRatings;
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
        this.itemRatings = buildItemRatings();
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
    
    @Override
    public double predictRating(int userId, int itemId) {
        if (!userRatings.containsKey(userId)) {
            return userAverages.getOrDefault(userId, 3.0);
        }
        
        // Find similar users who rated this item
        List<UserSimilarity> similarUsers = findSimilarUsers(userId, 20);
        
        double weightedSum = 0.0;
        double similaritySum = 0.0;
        
        for (UserSimilarity userSim : similarUsers) {
            int similarUserId = userSim.userId;
            double similarity = userSim.similarity;
            
            // Check if similar user rated this item
            Optional<Rating> rating = userRatings.get(similarUserId).stream()
                    .filter(r -> r.getItemId() == itemId)
                    .findFirst();
            
            if (rating.isPresent()) {
                double userAvg = userAverages.get(similarUserId);
                double adjustedRating = rating.get().getRating() - userAvg;
                
                weightedSum += similarity * adjustedRating;
                similaritySum += Math.abs(similarity);
            }
        }
        
        if (similaritySum == 0.0) {
            return userAverages.getOrDefault(userId, 3.0);
        }
        
        double userAvg = userAverages.get(userId);
        double predictedRating = userAvg + (weightedSum / similaritySum);
        
        return Math.max(1.0, Math.min(5.0, predictedRating));
    }
    
    @Override
    public double getUserSimilarity(int userId1, int userId2) {
        if (userId1 == userId2) {
            return 1.0;
        }
        
        String key = Math.min(userId1, userId2) + "_" + Math.max(userId1, userId2);
        return similarityCache.computeIfAbsent(key, k -> calculateSimilarity(userId1, userId2));
    }
    
    @Override
    public double getItemSimilarity(int itemId1, int itemId2) {
        // For user-based CF, we don't calculate item similarities
        return 0.0;
    }
    
    @Override
    public void addRating(Rating rating) {
        ratings.add(rating);
        updateDataStructures();
    }
    
    @Override
    public void removeRating(int userId, int itemId) {
        ratings.removeIf(r -> r.getUserId() == userId && r.getItemId() == itemId);
        updateDataStructures();
    }
    
    @Override
    public void updateRating(Rating rating) {
        removeRating(rating.getUserId(), rating.getItemId());
        addRating(rating);
    }
    
    @Override
    public List<Rating> getAllRatings() {
        return new ArrayList<>(ratings);
    }
    
    @Override
    public List<Rating> getUserRatings(int userId) {
        return userRatings.getOrDefault(userId, new ArrayList<>());
    }
    
    @Override
    public List<Rating> getItemRatings(int itemId) {
        return itemRatings.getOrDefault(itemId, new ArrayList<>());
    }
    
    @Override
    public boolean hasRating(int userId, int itemId) {
        return ratings.stream().anyMatch(r -> r.getUserId() == userId && r.getItemId() == itemId);
    }
    
    @Override
    public int getNumUsers() {
        return userRatings.size();
    }
    
    @Override
    public int getNumItems() {
        return itemRatings.size();
    }
    
    @Override
    public int getNumRatings() {
        return ratings.size();
    }
    
    /**
     * Find users similar to the target user.
     * 
     * @param userId the target user ID
     * @param maxUsers maximum number of similar users to return
     * @return list of similar users with their similarity scores
     */
    private List<UserSimilarity> findSimilarUsers(int userId, int maxUsers) {
        List<UserSimilarity> similarities = new ArrayList<>();
        
        for (int otherUserId : userRatings.keySet()) {
            if (otherUserId != userId) {
                double similarity = getUserSimilarity(userId, otherUserId);
                if (similarity > 0) {
                    similarities.add(new UserSimilarity(otherUserId, similarity));
                }
            }
        }
        
        similarities.sort(Collections.reverseOrder());
        return similarities.stream()
                .limit(maxUsers)
                .collect(Collectors.toList());
    }
    
    /**
     * Calculate similarity between two users.
     * 
     * @param userId1 first user ID
     * @param userId2 second user ID
     * @return similarity score
     */
    private double calculateSimilarity(int userId1, int userId2) {
        List<Rating> ratings1 = userRatings.get(userId1);
        List<Rating> ratings2 = userRatings.get(userId2);
        
        // Find common items
        Set<Integer> items1 = ratings1.stream().map(Rating::getItemId).collect(Collectors.toSet());
        Set<Integer> items2 = ratings2.stream().map(Rating::getItemId).collect(Collectors.toSet());
        Set<Integer> commonItems = new HashSet<>(items1);
        commonItems.retainAll(items2);
        
        if (commonItems.size() < minCommonItems) {
            return 0.0;
        }
        
        // Create rating vectors for common items
        Map<Integer, Double> ratingsMap1 = ratings1.stream()
                .filter(r -> commonItems.contains(r.getItemId()))
                .collect(Collectors.toMap(Rating::getItemId, Rating::getRating));
        
        Map<Integer, Double> ratingsMap2 = ratings2.stream()
                .filter(r -> commonItems.contains(r.getItemId()))
                .collect(Collectors.toMap(Rating::getItemId, Rating::getRating));
        
        List<Double> vector1 = new ArrayList<>();
        List<Double> vector2 = new ArrayList<>();
        
        for (int itemId : commonItems) {
            vector1.add(ratingsMap1.get(itemId));
            vector2.add(ratingsMap2.get(itemId));
        }
        
        // Calculate similarity based on chosen metric
        switch (similarityMetric) {
            case PEARSON:
                return calculatePearsonCorrelation(vector1, vector2);
            case COSINE:
                return calculateCosineSimilarity(vector1, vector2);
            case EUCLIDEAN:
                return calculateEuclideanSimilarity(vector1, vector2);
            default:
                return calculatePearsonCorrelation(vector1, vector2);
        }
    }
    
    /**
     * Calculate Pearson correlation coefficient.
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
    
    /**
     * Calculate cosine similarity.
     */
    private double calculateCosineSimilarity(List<Double> vector1, List<Double> vector2) {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        for (int i = 0; i < vector1.size(); i++) {
            double val1 = vector1.get(i);
            double val2 = vector2.get(i);
            dotProduct += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }
        
        double denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator == 0 ? 0 : dotProduct / denominator;
    }
    
    /**
     * Calculate Euclidean similarity (1 / (1 + distance)).
     */
    private double calculateEuclideanSimilarity(List<Double> vector1, List<Double> vector2) {
        double sumSquaredDiff = 0.0;
        
        for (int i = 0; i < vector1.size(); i++) {
            double diff = vector1.get(i) - vector2.get(i);
            sumSquaredDiff += diff * diff;
        }
        
        double distance = Math.sqrt(sumSquaredDiff);
        return 1.0 / (1.0 + distance);
    }
    
    /**
     * Build user ratings map.
     */
    private Map<Integer, List<Rating>> buildUserRatings() {
        return ratings.stream()
                .collect(Collectors.groupingBy(Rating::getUserId));
    }
    
    /**
     * Build item ratings map.
     */
    private Map<Integer, List<Rating>> buildItemRatings() {
        return ratings.stream()
                .collect(Collectors.groupingBy(Rating::getItemId));
    }
    
    /**
     * Calculate average rating for each user.
     */
    private Map<Integer, Double> calculateUserAverages() {
        Map<Integer, Double> averages = new HashMap<>();
        
        for (Map.Entry<Integer, List<Rating>> entry : userRatings.entrySet()) {
            int userId = entry.getKey();
            List<Rating> userRatings = entry.getValue();
            
            double avg = userRatings.stream()
                    .mapToDouble(Rating::getRating)
                    .average()
                    .orElse(3.0);
            
            averages.put(userId, avg);
        }
        
        return averages;
    }
    
    /**
     * Update data structures after changes.
     */
    private void updateDataStructures() {
        userRatings.clear();
        itemRatings.clear();
        userAverages.clear();
        similarityCache.clear();
        
        userRatings.putAll(buildUserRatings());
        itemRatings.putAll(buildItemRatings());
        userAverages.putAll(calculateUserAverages());
    }
    
    /**
     * Helper class for storing user similarity information.
     */
    private static class UserSimilarity implements Comparable<UserSimilarity> {
        final int userId;
        final double similarity;
        
        UserSimilarity(int userId, double similarity) {
            this.userId = userId;
            this.similarity = similarity;
        }
        
        @Override
        public int compareTo(UserSimilarity other) {
            return Double.compare(this.similarity, other.similarity);
        }
    }
}

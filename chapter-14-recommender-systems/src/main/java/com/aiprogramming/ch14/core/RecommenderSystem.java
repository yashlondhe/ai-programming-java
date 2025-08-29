package com.aiprogramming.ch14.core;

import java.util.List;

/**
 * Base interface for all recommender system implementations.
 * Defines the common contract that all recommender algorithms must follow.
 */
public interface RecommenderSystem {
    
    /**
     * Generate recommendations for a given user.
     * 
     * @param userId the ID of the user to generate recommendations for
     * @param numRecommendations the number of recommendations to generate
     * @return list of recommendation results sorted by relevance score
     */
    List<RecommendationResult> recommend(int userId, int numRecommendations);
    
    /**
     * Predict the rating a user would give to a specific item.
     * 
     * @param userId the ID of the user
     * @param itemId the ID of the item
     * @return predicted rating value
     */
    double predictRating(int userId, int itemId);
    
    /**
     * Get the similarity between two users.
     * 
     * @param userId1 the ID of the first user
     * @param userId2 the ID of the second user
     * @return similarity score between 0 and 1
     */
    double getUserSimilarity(int userId1, int userId2);
    
    /**
     * Get the similarity between two items.
     * 
     * @param itemId1 the ID of the first item
     * @param itemId2 the ID of the second item
     * @return similarity score between 0 and 1
     */
    double getItemSimilarity(int itemId1, int itemId2);
    
    /**
     * Add a new rating to the system.
     * 
     * @param rating the rating to add
     */
    void addRating(Rating rating);
    
    /**
     * Remove a rating from the system.
     * 
     * @param userId the ID of the user
     * @param itemId the ID of the item
     */
    void removeRating(int userId, int itemId);
    
    /**
     * Update an existing rating in the system.
     * 
     * @param rating the updated rating
     */
    void updateRating(Rating rating);
    
    /**
     * Get all ratings in the system.
     * 
     * @return list of all ratings
     */
    List<Rating> getAllRatings();
    
    /**
     * Get ratings for a specific user.
     * 
     * @param userId the ID of the user
     * @return list of ratings for the user
     */
    List<Rating> getUserRatings(int userId);
    
    /**
     * Get ratings for a specific item.
     * 
     * @param itemId the ID of the item
     * @return list of ratings for the item
     */
    List<Rating> getItemRatings(int itemId);
    
    /**
     * Check if a user has rated a specific item.
     * 
     * @param userId the ID of the user
     * @param itemId the ID of the item
     * @return true if the user has rated the item
     */
    boolean hasRating(int userId, int itemId);
    
    /**
     * Get the total number of users in the system.
     * 
     * @return number of users
     */
    int getNumUsers();
    
    /**
     * Get the total number of items in the system.
     * 
     * @return number of items
     */
    int getNumItems();
    
    /**
     * Get the total number of ratings in the system.
     * 
     * @return number of ratings
     */
    int getNumRatings();
}

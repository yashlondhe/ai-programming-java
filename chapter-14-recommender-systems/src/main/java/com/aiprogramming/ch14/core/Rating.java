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
     * Get the user ID.
     * 
     * @return user ID
     */
    public int getUserId() {
        return userId;
    }
    
    /**
     * Get the item ID.
     * 
     * @return item ID
     */
    public int getItemId() {
        return itemId;
    }
    
    /**
     * Get the rating value.
     * 
     * @return rating value
     */
    public double getRating() {
        return rating;
    }
    
    /**
     * Get the timestamp when the rating was given.
     * 
     * @return timestamp
     */
    public LocalDateTime getTimestamp() {
        return timestamp;
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
     * Check if this rating is positive (above 3.5).
     * 
     * @return true if rating is above 3.5
     */
    public boolean isPositive() {
        return isPositive(3.5);
    }
    
    /**
     * Get the normalized rating (0-1 scale).
     * 
     * @return normalized rating
     */
    public double getNormalizedRating() {
        return rating / 5.0;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Rating rating1 = (Rating) obj;
        return userId == rating1.userId && 
               itemId == rating1.itemId && 
               Double.compare(rating1.rating, rating) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(userId, itemId, rating);
    }
    
    @Override
    public String toString() {
        return String.format("Rating{userId=%d, itemId=%d, rating=%.1f, timestamp=%s}", 
                           userId, itemId, rating, timestamp);
    }
}

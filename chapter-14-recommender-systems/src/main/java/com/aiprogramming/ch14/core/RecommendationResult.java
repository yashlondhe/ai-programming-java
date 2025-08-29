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
     * Get the recommended item ID.
     * 
     * @return item ID
     */
    public Object getItemId() {
        return itemId;
    }
    
    /**
     * Get the relevance score.
     * 
     * @return relevance score
     */
    public double getScore() {
        return score;
    }
    
    /**
     * Get the explanation for this recommendation.
     * 
     * @return explanation string
     */
    public String getReason() {
        return reason;
    }
    
    /**
     * Check if this recommendation has an explanation.
     * 
     * @return true if there is an explanation
     */
    public boolean hasReason() {
        return !reason.isEmpty();
    }
    
    /**
     * Get the normalized score (0-1 scale).
     * 
     * @param maxScore the maximum possible score for normalization
     * @return normalized score
     */
    public double getNormalizedScore(double maxScore) {
        return maxScore > 0 ? score / maxScore : 0.0;
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
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        RecommendationResult that = (RecommendationResult) obj;
        return itemId == that.itemId && 
               Double.compare(that.score, score) == 0 &&
               Objects.equals(reason, that.reason);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(itemId, score, reason);
    }
    
    @Override
    public String toString() {
        if (hasReason()) {
            return String.format("Recommendation{itemId=%d, score=%.3f, reason='%s'}", 
                               itemId, score, reason);
        } else {
            return String.format("Recommendation{itemId=%d, score=%.3f}", itemId, score);
        }
    }
}

package com.aiprogramming.ch14.content;

import com.aiprogramming.ch14.core.RecommendationResult;
import com.aiprogramming.ch14.core.Rating;

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
     * Add items with their ratings to calculate averages.
     * 
     * @param ratings list of ratings
     */
    public void addRatings(List<Rating> ratings) {
        Map<String, List<Rating>> itemRatings = ratings.stream()
                .collect(Collectors.groupingBy(r -> String.valueOf(r.getItemId())));
        
        for (Map.Entry<String, List<Rating>> entry : itemRatings.entrySet()) {
            String itemId = entry.getKey();
            List<Rating> itemRatingList = entry.getValue();
            
            double avg = itemRatingList.stream()
                    .mapToDouble(Rating::getRating)
                    .average()
                    .orElse(3.0);
            
            itemAverages.put(itemId, avg);
        }
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
    
    /**
     * Generate recommendations based on user's liked items.
     * 
     * @param likedItemIds list of item IDs the user has liked
     * @param numRecommendations number of recommendations to generate
     * @return list of recommendations
     */
    public List<RecommendationResult> recommendFromLikedItems(List<String> likedItemIds, int numRecommendations) {
        if (likedItemIds.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Create user profile from liked items
        FeatureVector userProfile = createUserProfileFromLikedItems(likedItemIds);
        return recommend(userProfile, numRecommendations);
    }
    
    /**
     * Create a user profile from liked items.
     * 
     * @param likedItemIds list of liked item IDs
     * @return user profile feature vector
     */
    private FeatureVector createUserProfileFromLikedItems(List<String> likedItemIds) {
        Map<String, Double> aggregatedFeatures = new HashMap<>();
        Map<String, String> aggregatedCategoricalFeatures = new HashMap<>();
        
        int itemCount = 0;
        for (String itemId : likedItemIds) {
            FeatureVector itemFeatures = items.get(itemId);
            if (itemFeatures != null) {
                itemCount++;
                
                // Aggregate numerical features
                for (String featureName : itemFeatures.getNumericalFeatureNames()) {
                    double value = itemFeatures.getFeature(featureName);
                    aggregatedFeatures.merge(featureName, value, Double::sum);
                }
                
                // Aggregate categorical features
                for (String featureName : itemFeatures.getCategoricalFeatureNames()) {
                    String value = itemFeatures.getCategoricalFeature(featureName);
                    if (value != null) {
                        aggregatedCategoricalFeatures.put(featureName, value);
                    }
                }
            }
        }
        
        // Average the numerical features
        if (itemCount > 0) {
            for (String featureName : aggregatedFeatures.keySet()) {
                aggregatedFeatures.put(featureName, aggregatedFeatures.get(featureName) / itemCount);
            }
        }
        
        return new FeatureVector("user_profile", aggregatedFeatures, aggregatedCategoricalFeatures);
    }
    
    /**
     * Get similar items to a given item.
     * 
     * @param itemId the item ID to find similar items for
     * @param numSimilarItems number of similar items to return
     * @return list of similar items with similarity scores
     */
    public List<RecommendationResult> getSimilarItems(String itemId, int numSimilarItems) {
        FeatureVector targetItem = items.get(itemId);
        if (targetItem == null) {
            return new ArrayList<>();
        }
        
        List<RecommendationResult> similarItems = new ArrayList<>();
        
        for (FeatureVector itemFeatures : items.values()) {
            if (!itemFeatures.getItemId().equals(itemId)) {
                double similarity = targetItem.combinedSimilarity(itemFeatures, 0.5);
                if (similarity > 0) {
                    similarItems.add(new RecommendationResult(itemFeatures.getItemId(), similarity));
                }
            }
        }
        
        similarItems.sort(Collections.reverseOrder());
        return similarItems.stream()
                .limit(numSimilarItems)
                .collect(Collectors.toList());
    }
    
    /**
     * Get the feature vector for an item.
     * 
     * @param itemId the item ID
     * @return feature vector, or null if not found
     */
    public FeatureVector getItemFeatures(String itemId) {
        return items.get(itemId);
    }
    
    /**
     * Get all item IDs in the system.
     * 
     * @return set of item IDs
     */
    public Set<String> getAllItemIds() {
        return new HashSet<>(items.keySet());
    }
    
    /**
     * Get the number of items in the system.
     * 
     * @return number of items
     */
    public int getNumItems() {
        return items.size();
    }
    
    /**
     * Remove an item from the system.
     * 
     * @param itemId the item ID to remove
     */
    public void removeItem(String itemId) {
        items.remove(itemId);
        itemAverages.remove(itemId);
    }
    
    /**
     * Update an item's features.
     * 
     * @param itemFeatures the updated feature vector
     */
    public void updateItem(FeatureVector itemFeatures) {
        items.put(itemFeatures.getItemId(), itemFeatures);
    }
    
    /**
     * Get items that have a specific feature.
     * 
     * @param featureName the feature name to search for
     * @return list of item IDs that have this feature
     */
    public List<String> getItemsWithFeature(String featureName) {
        return items.values().stream()
                .filter(item -> item.hasFeature(featureName))
                .map(FeatureVector::getItemId)
                .collect(Collectors.toList());
    }
    
    /**
     * Get items that have a specific categorical feature value.
     * 
     * @param featureName the feature name
     * @param value the feature value
     * @return list of item IDs that have this feature value
     */
    public List<String> getItemsWithCategoricalFeature(String featureName, String value) {
        return items.values().stream()
                .filter(item -> item.hasCategoricalFeature(featureName, value))
                .map(FeatureVector::getItemId)
                .collect(Collectors.toList());
    }
}

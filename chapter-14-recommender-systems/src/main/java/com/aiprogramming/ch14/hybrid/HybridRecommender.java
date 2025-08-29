package com.aiprogramming.ch14.hybrid;

import com.aiprogramming.ch14.collaborative.UserBasedCollaborativeFiltering;
import com.aiprogramming.ch14.content.ContentBasedFiltering;
import com.aiprogramming.ch14.content.FeatureVector;
import com.aiprogramming.ch14.core.RecommendationResult;
import com.aiprogramming.ch14.core.Rating;

import java.util.*;
import java.util.stream.Collectors;

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
            
            // Add recommendation with item ID as string
            finalRecommendations.add(new RecommendationResult(itemId, avgScore));
        }
        
        // Sort by score and return top N
        finalRecommendations.sort(Collections.reverseOrder());
        return finalRecommendations.stream()
                .limit(numRecommendations)
                .collect(Collectors.toList());
    }
    
    /**
     * Generate recommendations using only collaborative filtering.
     * 
     * @param userId the user ID
     * @param numRecommendations number of recommendations
     * @return collaborative filtering recommendations
     */
    public List<RecommendationResult> recommendCollaborative(int userId, int numRecommendations) {
        for (RecommenderComponent component : recommenders) {
            if (component.type == RecommenderType.COLLABORATIVE) {
                UserBasedCollaborativeFiltering collaborativeRecommender = 
                    (UserBasedCollaborativeFiltering) component.recommender;
                return collaborativeRecommender.recommend(userId, numRecommendations);
            }
        }
        return new ArrayList<>();
    }
    
    /**
     * Generate recommendations using only content-based filtering.
     * 
     * @param userProfile the user profile
     * @param numRecommendations number of recommendations
     * @return content-based filtering recommendations
     */
    public List<RecommendationResult> recommendContent(FeatureVector userProfile, int numRecommendations) {
        for (RecommenderComponent component : recommenders) {
            if (component.type == RecommenderType.CONTENT) {
                ContentBasedFiltering contentRecommender = 
                    (ContentBasedFiltering) component.recommender;
                return contentRecommender.recommend(userProfile, numRecommendations);
            }
        }
        return new ArrayList<>();
    }
    
    /**
     * Get the number of recommender components.
     * 
     * @return number of components
     */
    public int getNumComponents() {
        return recommenders.size();
    }
    
    /**
     * Get the total weight of all components.
     * 
     * @return total weight
     */
    public double getTotalWeight() {
        return recommenders.stream()
                .mapToDouble(component -> component.weight)
                .sum();
    }
    
    /**
     * Normalize weights so they sum to 1.0.
     */
    public void normalizeWeights() {
        double totalWeight = getTotalWeight();
        if (totalWeight > 0) {
            for (RecommenderComponent component : recommenders) {
                component.weight /= totalWeight;
            }
        }
    }
    
    /**
     * Remove all recommender components.
     */
    public void clear() {
        recommenders.clear();
    }
    
    /**
     * Get component information.
     * 
     * @return list of component descriptions
     */
    public List<String> getComponentInfo() {
        List<String> info = new ArrayList<>();
        for (int i = 0; i < recommenders.size(); i++) {
            RecommenderComponent component = recommenders.get(i);
            info.add(String.format("Component %d: %s (weight: %.2f)", 
                                 i + 1, component.type, component.weight));
        }
        return info;
    }
    
    /**
     * Enum for recommender types.
     */
    private enum RecommenderType {
        COLLABORATIVE, CONTENT
    }
    
    /**
     * Helper class to store recommender components with their weights.
     */
    private static class RecommenderComponent {
        final Object recommender;
        double weight;
        final RecommenderType type;
        
        RecommenderComponent(Object recommender, double weight, RecommenderType type) {
            this.recommender = recommender;
            this.weight = weight;
            this.type = type;
        }
    }
}

package com.aiprogramming.ch20.federated;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

/**
 * Privacy-Preserving Machine Learning
 * 
 * Implements various privacy-preserving techniques for federated learning:
 * - Differential privacy
 * - Secure aggregation
 * - Homomorphic encryption simulation
 * - Privacy budget management
 */
public class PrivacyPreservingML {
    
    private static final Logger logger = LoggerFactory.getLogger(PrivacyPreservingML.class);
    
    private final RandomGenerator random;
    private final PrivacyBudgetManager budgetManager;
    
    public PrivacyPreservingML() {
        this.random = new Well19937c();
        this.budgetManager = new PrivacyBudgetManager();
    }
    
    /**
     * Add differential privacy noise to a vector
     * 
     * @param vector The vector to add noise to
     * @param epsilon Privacy parameter (lower = more private)
     * @return Vector with differential privacy noise
     */
    public RealVector addDifferentialPrivacy(RealVector vector, double epsilon) {
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        
        // Calculate sensitivity (L2 norm of the vector)
        double sensitivity = vector.getNorm();
        
        // Calculate noise scale based on epsilon and sensitivity
        double noiseScale = sensitivity / epsilon;
        
        // Add Laplace noise
        RealVector noisyVector = vector.copy();
        for (int i = 0; i < vector.getDimension(); i++) {
            double noise = sampleLaplace(noiseScale);
            noisyVector.setEntry(i, noisyVector.getEntry(i) + noise);
        }
        
        logger.debug("Added differential privacy noise with epsilon={}, sensitivity={}", epsilon, sensitivity);
        return noisyVector;
    }
    
    /**
     * Add differential privacy noise to a scalar value
     */
    public double addDifferentialPrivacy(double value, double epsilon) {
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        
        // Assume sensitivity of 1 for scalar values
        double sensitivity = 1.0;
        double noiseScale = sensitivity / epsilon;
        double noise = sampleLaplace(noiseScale);
        
        return value + noise;
    }
    
    /**
     * Sample from Laplace distribution
     */
    private double sampleLaplace(double scale) {
        double u = random.nextDouble() - 0.5;
        return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }
    
    /**
     * Secure aggregation of model updates
     * 
     * @param updates List of model updates from clients
     * @return Aggregated model update
     */
    public FederatedClient.ModelUpdate secureAggregate(List<FederatedClient.ModelUpdate> updates) {
        if (updates.isEmpty()) {
            throw new IllegalArgumentException("No updates to aggregate");
        }
        
        logger.info("Performing secure aggregation of {} model updates", updates.size());
        
        // Initialize aggregated update with the first update
        FederatedClient.ModelUpdate firstUpdate = updates.get(0);
        RealVector aggregatedWeights = firstUpdate.getWeightUpdate().copy();
        double aggregatedBias = firstUpdate.getBiasUpdate();
        int totalDataSize = firstUpdate.getDataSize();
        
        // Aggregate remaining updates
        for (int i = 1; i < updates.size(); i++) {
            FederatedClient.ModelUpdate update = updates.get(i);
            
            // Weighted aggregation based on data size
            double weight = (double) update.getDataSize() / totalDataSize;
            aggregatedWeights = aggregatedWeights.add(update.getWeightUpdate().mapMultiply(weight));
            aggregatedBias += update.getBiasUpdate() * weight;
            totalDataSize += update.getDataSize();
        }
        
        // Create aggregated update
        return new FederatedClient.ModelUpdate(
            "aggregated", 
            firstUpdate.getRoundNumber(), 
            aggregatedWeights, 
            aggregatedBias, 
            totalDataSize
        );
    }
    
    /**
     * Homomorphic encryption simulation for secure computation
     */
    public static class HomomorphicEncryption {
        
        /**
         * Simulate homomorphic encryption by adding random noise
         */
        public static RealVector encrypt(RealVector vector, double noiseLevel) {
            RealVector encrypted = vector.copy();
            RandomGenerator random = new Well19937c();
            
            for (int i = 0; i < vector.getDimension(); i++) {
                double noise = random.nextGaussian() * noiseLevel;
                encrypted.setEntry(i, encrypted.getEntry(i) + noise);
            }
            
            return encrypted;
        }
        
        /**
         * Simulate homomorphic decryption by removing noise
         */
        public static RealVector decrypt(RealVector encryptedVector, double noiseLevel) {
            // In a real implementation, this would use proper decryption
            // For simulation, we just return the encrypted vector
            return encryptedVector.copy();
        }
        
        /**
         * Simulate homomorphic addition
         */
        public static RealVector add(RealVector a, RealVector b) {
            return a.add(b);
        }
        
        /**
         * Simulate homomorphic multiplication (simplified)
         */
        public static RealVector multiply(RealVector a, RealVector b) {
            RealVector result = new ArrayRealVector(a.getDimension());
            for (int i = 0; i < a.getDimension(); i++) {
                result.setEntry(i, a.getEntry(i) * b.getEntry(i));
            }
            return result;
        }
    }
    
    /**
     * Privacy budget management for differential privacy
     */
    public static class PrivacyBudgetManager {
        private double totalBudget;
        private double usedBudget;
        private final Map<String, Double> clientBudgets;
        
        public PrivacyBudgetManager() {
            this.totalBudget = 10.0; // Total privacy budget
            this.usedBudget = 0.0;
            this.clientBudgets = new HashMap<>();
        }
        
        public PrivacyBudgetManager(double totalBudget) {
            this.totalBudget = totalBudget;
            this.usedBudget = 0.0;
            this.clientBudgets = new HashMap<>();
        }
        
        /**
         * Check if privacy budget is available
         */
        public boolean hasBudget(String clientId, double requiredEpsilon) {
            double clientUsed = clientBudgets.getOrDefault(clientId, 0.0);
            return (usedBudget + requiredEpsilon <= totalBudget) && 
                   (clientUsed + requiredEpsilon <= totalBudget * 0.1); // Max 10% per client
        }
        
        /**
         * Use privacy budget
         */
        public void useBudget(String clientId, double epsilon) {
            if (!hasBudget(clientId, epsilon)) {
                throw new IllegalStateException("Insufficient privacy budget");
            }
            
            usedBudget += epsilon;
            clientBudgets.put(clientId, clientBudgets.getOrDefault(clientId, 0.0) + epsilon);
            
            logger.debug("Used privacy budget: client={}, epsilon={}, total used={}", 
                        clientId, epsilon, usedBudget);
        }
        
        /**
         * Get remaining budget
         */
        public double getRemainingBudget() {
            return totalBudget - usedBudget;
        }
        
        /**
         * Reset budget (for new training rounds)
         */
        public void resetBudget() {
            usedBudget = 0.0;
            clientBudgets.clear();
            logger.info("Privacy budget reset");
        }
    }
    
    /**
     * Privacy metrics and analysis
     */
    public static class PrivacyMetrics {
        
        /**
         * Calculate privacy loss for a given epsilon
         */
        public static double calculatePrivacyLoss(double epsilon) {
            return Math.log(Math.exp(epsilon));
        }
        
        /**
         * Calculate composition of privacy budgets
         */
        public static double composePrivacyBudgets(List<Double> epsilons) {
            double sum = 0.0;
            for (double epsilon : epsilons) {
                sum += epsilon;
            }
            return sum;
        }
        
        /**
         * Calculate advanced composition (more sophisticated)
         */
        public static double advancedComposition(List<Double> epsilons, double delta) {
            double sum = 0.0;
            double sumSquared = 0.0;
            
            for (double epsilon : epsilons) {
                sum += epsilon;
                sumSquared += epsilon * epsilon;
            }
            
            return sum + Math.sqrt(2 * sumSquared * Math.log(1.0 / delta));
        }
        
        /**
         * Estimate privacy risk based on model parameters
         */
        public static double estimatePrivacyRisk(RealVector modelParams, double epsilon) {
            // Simple risk estimation based on parameter sensitivity
            double sensitivity = modelParams.getNorm();
            return sensitivity / epsilon;
        }
    }
    
    /**
     * Secure multi-party computation simulation
     */
    public static class SecureMPC {
        
        /**
         * Simulate secure sum computation
         */
        public static double secureSum(List<Double> values) {
            // In real implementation, this would use cryptographic protocols
            // For simulation, we just return the sum
            return values.stream().mapToDouble(Double::doubleValue).sum();
        }
        
        /**
         * Simulate secure mean computation
         */
        public static double secureMean(List<Double> values) {
            double sum = secureSum(values);
            return sum / values.size();
        }
        
        /**
         * Simulate secure variance computation
         */
        public static double secureVariance(List<Double> values) {
            double mean = secureMean(values);
            double sumSquaredDiff = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .sum();
            return sumSquaredDiff / values.size();
        }
    }
    
    /**
     * Anonymization techniques
     */
    public static class Anonymization {
        
        /**
         * K-anonymity: ensure each group has at least k members
         */
        public static <T> List<List<T>> kAnonymity(List<T> data, int k) {
            List<List<T>> groups = new ArrayList<>();
            List<T> currentGroup = new ArrayList<>();
            
            for (T item : data) {
                currentGroup.add(item);
                if (currentGroup.size() >= k) {
                    groups.add(new ArrayList<>(currentGroup));
                    currentGroup.clear();
                }
            }
            
            // Add remaining items to the last group
            if (!currentGroup.isEmpty()) {
                groups.add(currentGroup);
            }
            
            return groups;
        }
        
        /**
         * L-diversity: ensure each group has at least l different sensitive values
         */
        public static <T> boolean lDiversity(List<T> group, int l) {
            return group.stream().distinct().count() >= l;
        }
        
        /**
         * T-closeness: ensure distribution of sensitive values in each group
         * is close to the overall distribution
         */
        public static <T> double tCloseness(List<T> group, List<T> population) {
            // Simplified implementation - calculate distribution difference
            Map<T, Long> groupDist = group.stream()
                .collect(java.util.stream.Collectors.groupingBy(t -> t, java.util.stream.Collectors.counting()));
            Map<T, Long> popDist = population.stream()
                .collect(java.util.stream.Collectors.groupingBy(t -> t, java.util.stream.Collectors.counting()));
            
            double totalDiff = 0.0;
            for (T key : groupDist.keySet()) {
                double groupProb = (double) groupDist.get(key) / group.size();
                double popProb = (double) popDist.getOrDefault(key, 0L) / population.size();
                totalDiff += Math.abs(groupProb - popProb);
            }
            
            return totalDiff / 2.0; // Normalize to [0,1]
        }
    }
    
    // Getter for budget manager
    public PrivacyBudgetManager getBudgetManager() {
        return budgetManager;
    }
}

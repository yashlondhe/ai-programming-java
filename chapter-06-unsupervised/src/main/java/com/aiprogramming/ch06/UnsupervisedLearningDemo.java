package com.aiprogramming.ch06;

import java.util.*;
import com.aiprogramming.utils.*;

/**
 * Main demo class showcasing unsupervised learning algorithms
 */
public class UnsupervisedLearningDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 6: Unsupervised Learning Demo ===\n");
        
        // Demo 1: Clustering Algorithms
        demonstrateClustering();
        
        // Demo 2: Dimensionality Reduction
        demonstrateDimensionalityReduction();
        
        // Demo 3: Association Rule Learning
        demonstrateAssociationRules();
        
        System.out.println("\n=== Demo Complete ===");
    }
    
    /**
     * Demonstrate clustering algorithms (K-Means and DBSCAN)
     */
    private static void demonstrateClustering() {
        System.out.println("1. CLUSTERING ALGORITHMS");
        System.out.println("========================");
        
        // Generate sample data
        List<DataPoint> dataPoints = generateSampleData(300);
        System.out.printf("Generated %d data points for clustering\n", dataPoints.size());
        
        // Validate data using utils
        ValidationUtils.validateNotNull(dataPoints, "dataPoints");
        ValidationUtils.validateNonEmpty(dataPoints, "dataPoints");
        
        // K-Means Clustering
        System.out.println("\n--- K-Means Clustering ---");
        KMeans kmeans = new KMeans(3);
        ClusteringResult kmeansResult = kmeans.fit(dataPoints);
        kmeansResult.printStatistics();
        
        // Find optimal k using elbow method
        System.out.println("\nFinding optimal k using elbow method:");
        int optimalK = KMeans.findOptimalK(dataPoints, 10);
        System.out.printf("Optimal k: %d\n", optimalK);
        
        // DBSCAN Clustering
        System.out.println("\n--- DBSCAN Clustering ---");
        double epsilon = DBSCAN.findOptimalEpsilon(dataPoints, 5);
        System.out.printf("Optimal epsilon: %.3f\n", epsilon);
        
        DBSCAN dbscan = new DBSCAN(epsilon, 5);
        ClusteringResult dbscanResult = dbscan.fit(dataPoints);
        dbscanResult.printStatistics();
        
        // Compare clustering results using utils
        System.out.println("\n--- Clustering Comparison ---");
        System.out.printf("K-Means Silhouette Score: %.4f\n", kmeansResult.getSilhouetteScore());
        System.out.printf("DBSCAN Silhouette Score: %.4f\n", dbscanResult.getSilhouetteScore());
        System.out.printf("K-Means Inertia: %.4f\n", kmeansResult.getInertia());
        System.out.printf("DBSCAN Inertia: %.4f\n", dbscanResult.getInertia());
        
        // Calculate statistics using utils
        double[] silhouetteScores = {kmeansResult.getSilhouetteScore(), dbscanResult.getSilhouetteScore()};
        System.out.printf("Average silhouette score: %.4f\n", StatisticsUtils.mean(silhouetteScores));
        System.out.printf("Silhouette score variance: %.4f\n", StatisticsUtils.variance(silhouetteScores));
    }
    
    /**
     * Demonstrate dimensionality reduction with PCA
     */
    private static void demonstrateDimensionalityReduction() {
        System.out.println("\n\n2. DIMENSIONALITY REDUCTION");
        System.out.println("===========================");
        
        // Generate high-dimensional data
        List<DataPoint> highDimData = generateHighDimensionalData(100, 10);
        System.out.printf("Generated %d data points with %d dimensions\n", 
                         highDimData.size(), highDimData.get(0).getDimension());
        
        // Validate data using utils
        ValidationUtils.validateNotNull(highDimData, "highDimData");
        ValidationUtils.validateNonEmpty(highDimData, "highDimData");
        
        // Apply PCA
        System.out.println("\n--- PCA Dimensionality Reduction ---");
        PCA pca = new PCA(2);
        List<DataPoint> reducedData = pca.fitTransform(highDimData);
        
        System.out.printf("Reduced data from %d to %d dimensions\n", 
                         highDimData.get(0).getDimension(), reducedData.get(0).getDimension());
        
        // Show explained variance ratio
        List<Double> explainedVariance = pca.getExplainedVarianceRatio();
        System.out.println("Explained variance ratio:");
        for (int i = 0; i < explainedVariance.size(); i++) {
            System.out.printf("  PC%d: %.3f\n", i + 1, explainedVariance.get(i));
        }
        
        // Calculate cumulative explained variance using utils
        double[] varianceArray = explainedVariance.stream().mapToDouble(Double::doubleValue).toArray();
        double cumulativeVariance = 0.0;
        for (double v : varianceArray) {
            cumulativeVariance += v;
        }
        System.out.printf("Cumulative explained variance: %.3f\n", cumulativeVariance);
        
        // Demonstrate inverse transform
        System.out.println("\n--- PCA Inverse Transform ---");
        List<DataPoint> reconstructedData = pca.inverseTransform(reducedData);
        
        // Calculate reconstruction error using utils
        double reconstructionError = calculateReconstructionError(highDimData, reconstructedData);
        System.out.printf("Reconstruction error: %.6f\n", reconstructionError);
        
        // Show sample transformations
        System.out.println("\nSample transformations:");
        for (int i = 0; i < Math.min(3, highDimData.size()); i++) {
            System.out.printf("Original: %s\n", highDimData.get(i).getFeatures().subList(0, 3));
            System.out.printf("Reduced:  %s\n", reducedData.get(i).getFeatures());
            System.out.printf("Reconstructed: %s\n", reconstructedData.get(i).getFeatures().subList(0, 3));
            System.out.println();
        }
    }
    
    /**
     * Demonstrate association rule learning with Apriori
     */
    private static void demonstrateAssociationRules() {
        System.out.println("\n\n3. ASSOCIATION RULE LEARNING");
        System.out.println("============================");
        
        // Generate sample transaction data
        List<Set<String>> transactions = generateTransactionData(1000);
        System.out.printf("Generated %d transactions\n", transactions.size());
        
        // Validate data using utils
        ValidationUtils.validateNotNull(transactions, "transactions");
        ValidationUtils.validateNonEmpty(transactions, "transactions");
        
        // Show sample transactions
        System.out.println("\nSample transactions:");
        for (int i = 0; i < Math.min(5, transactions.size()); i++) {
            System.out.printf("Transaction %d: %s\n", i + 1, transactions.get(i));
        }
        
        // Apply Apriori algorithm
        System.out.println("\n--- Apriori Association Rule Mining ---");
        Apriori apriori = new Apriori(0.1, 0.5); // 10% support, 50% confidence
        apriori.fit(transactions);
        
        // Show frequent itemsets
        List<Set<String>> frequentItemsets = apriori.getFrequentItemsets();
        System.out.printf("\nFound %d frequent itemsets:\n", frequentItemsets.size());
        for (int i = 0; i < Math.min(10, frequentItemsets.size()); i++) {
            System.out.printf("  %s\n", frequentItemsets.get(i));
        }
        
        // Show association rules
        List<Apriori.AssociationRule> rules = apriori.getAssociationRules();
        System.out.printf("\nFound %d association rules:\n", rules.size());
        for (int i = 0; i < Math.min(10, rules.size()); i++) {
            System.out.printf("  %s\n", rules.get(i));
        }
        
        // Analyze rule statistics using utils
        if (!rules.isEmpty()) {
            double[] confidences = rules.stream()
                    .mapToDouble(Apriori.AssociationRule::getConfidence)
                    .toArray();
            
            System.out.printf("\nConfidence Statistics:\n");
            System.out.printf("  Average confidence: %.3f\n", StatisticsUtils.mean(confidences));
            
            // Calculate max and min manually
            double maxConfidence = confidences[0];
            double minConfidence = confidences[0];
            for (double conf : confidences) {
                if (conf > maxConfidence) maxConfidence = conf;
                if (conf < minConfidence) minConfidence = conf;
            }
            System.out.printf("  Maximum confidence: %.3f\n", maxConfidence);
            System.out.printf("  Minimum confidence: %.3f\n", minConfidence);
            System.out.printf("  Confidence variance: %.3f\n", StatisticsUtils.variance(confidences));
            System.out.printf("  Confidence standard deviation: %.3f\n", StatisticsUtils.standardDeviation(confidences));
        }
    }
    
    /**
     * Generate sample data for clustering
     */
    private static List<DataPoint> generateSampleData(int numPoints) {
        List<DataPoint> dataPoints = new ArrayList<>();
        Random random = new Random(42);
        
        // Generate 3 clusters
        for (int i = 0; i < numPoints; i++) {
            List<Double> features = new ArrayList<>();
            
            if (i < numPoints / 3) {
                // Cluster 1: centered around (2, 2)
                features.add(2 + random.nextGaussian() * 0.5);
                features.add(2 + random.nextGaussian() * 0.5);
            } else if (i < 2 * numPoints / 3) {
                // Cluster 2: centered around (8, 8)
                features.add(8 + random.nextGaussian() * 0.5);
                features.add(8 + random.nextGaussian() * 0.5);
            } else {
                // Cluster 3: centered around (5, 2)
                features.add(5 + random.nextGaussian() * 0.5);
                features.add(2 + random.nextGaussian() * 0.5);
            }
            
            dataPoints.add(new DataPoint(features));
        }
        
        return dataPoints;
    }
    
    /**
     * Generate high-dimensional data for PCA
     */
    private static List<DataPoint> generateHighDimensionalData(int numPoints, int dimensions) {
        List<DataPoint> dataPoints = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numPoints; i++) {
            List<Double> features = new ArrayList<>();
            
            // Generate correlated features
            double baseValue = random.nextGaussian();
            for (int j = 0; j < dimensions; j++) {
                features.add(baseValue + random.nextGaussian() * 0.1);
            }
            
            dataPoints.add(new DataPoint(features));
        }
        
        return dataPoints;
    }
    
    /**
     * Generate transaction data for association rule learning
     */
    private static List<Set<String>> generateTransactionData(int numTransactions) {
        List<Set<String>> transactions = new ArrayList<>();
        Random random = new Random(42);
        
        String[] items = {"bread", "milk", "eggs", "butter", "cheese", "yogurt", "apples", "bananas", "oranges"};
        
        for (int i = 0; i < numTransactions; i++) {
            Set<String> transaction = new HashSet<>();
            
            // Add 2-5 random items per transaction
            int numItems = 2 + random.nextInt(4);
            for (int j = 0; j < numItems; j++) {
                transaction.add(items[random.nextInt(items.length)]);
            }
            
            transactions.add(transaction);
        }
        
        return transactions;
    }
    
    /**
     * Calculate reconstruction error between original and reconstructed data
     */
    private static double calculateReconstructionError(List<DataPoint> original, List<DataPoint> reconstructed) {
        ValidationUtils.validateNotNull(original, "original");
        ValidationUtils.validateNotNull(reconstructed, "reconstructed");
        if (original.size() != reconstructed.size()) {
            throw new IllegalArgumentException("Data sets must have the same size");
        }
        
        double totalError = 0.0;
        int totalFeatures = 0;
        
        for (int i = 0; i < original.size(); i++) {
            DataPoint orig = original.get(i);
            DataPoint recon = reconstructed.get(i);
            
            for (int j = 0; j < orig.getDimension(); j++) {
                double diff = orig.getFeature(j) - recon.getFeature(j);
                totalError += diff * diff;
                totalFeatures++;
            }
        }
        
        // Use utils for final calculation
        return Math.sqrt(totalError / totalFeatures);
    }
}

# Chapter 6: Unsupervised Learning

## Introduction

Unsupervised learning is a type of machine learning where the algorithm discovers patterns in data without being given labeled examples. Unlike supervised learning, where we have input-output pairs, unsupervised learning works with unlabeled data to find hidden structures, groupings, or relationships.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of unsupervised learning
- Implement clustering algorithms to group similar data points
- Apply dimensionality reduction techniques to simplify complex data
- Discover association rules in transactional data
- Evaluate the performance of unsupervised learning algorithms
- Choose appropriate algorithms for different types of problems

### Key Concepts

- **Clustering**: Grouping similar data points together
- **Dimensionality Reduction**: Reducing the number of features while preserving important information
- **Association Rule Learning**: Finding relationships between items in large datasets
- **Evaluation Metrics**: Measuring the quality of unsupervised learning results

## 6.1 Clustering Algorithms

Clustering is the process of grouping similar data points together. The goal is to find natural groupings in the data without any prior knowledge of what these groups should be.

### 6.1.1 K-Means Clustering

K-Means is one of the most popular clustering algorithms. It partitions data into k clusters, where each data point belongs to the cluster with the nearest mean.

#### Algorithm Overview

1. **Initialization**: Choose k initial centroids randomly
2. **Assignment**: Assign each data point to the nearest centroid
3. **Update**: Recalculate centroids as the mean of all points in each cluster
4. **Repeat**: Steps 2-3 until convergence

#### Implementation

```java
package com.aiprogramming.ch06;

import java.util.*;
import java.util.stream.Collectors;

/**
 * K-Means clustering algorithm implementation
 */
public class KMeans {
    private final int k;
    private final int maxIterations;
    private final double tolerance;
    private List<DataPoint> centroids;
    private boolean trained;
    
    public KMeans(int k) {
        this(k, 100, 1e-4);
    }
    
    public KMeans(int k, int maxIterations, double tolerance) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive");
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("Max iterations must be positive");
        }
        if (tolerance <= 0) {
            throw new IllegalArgumentException("Tolerance must be positive");
        }
        
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.centroids = new ArrayList<>();
        this.trained = false;
    }
    
    /**
     * Train the K-Means model
     */
    public ClusteringResult fit(List<DataPoint> dataPoints) {
        if (dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points cannot be empty");
        }
        
        if (dataPoints.size() < k) {
            throw new IllegalArgumentException("Number of data points must be >= k");
        }
        
        // Initialize centroids using k-means++
        initializeCentroids(dataPoints);
        
        // Main K-Means loop
        double previousInertia = Double.MAX_VALUE;
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Assign points to nearest centroid
            assignToClusters(dataPoints);
            
            // Update centroids
            updateCentroids(dataPoints);
            
            // Check convergence
            double currentInertia = calculateInertia(dataPoints);
            if (Math.abs(previousInertia - currentInertia) < tolerance) {
                break;
            }
            previousInertia = currentInertia;
        }
        
        this.trained = true;
        
        // Create clustering result
        Map<Integer, List<DataPoint>> clusters = new HashMap<>();
        for (int i = 0; i < k; i++) {
            clusters.put(i, new ArrayList<>());
        }
        
        for (DataPoint point : dataPoints) {
            clusters.get(point.getClusterId()).add(point);
        }
        
        return new ClusteringResult(dataPoints, clusters);
    }
    
    /**
     * Initialize centroids using k-means++ algorithm
     */
    private void initializeCentroids(List<DataPoint> dataPoints) {
        centroids.clear();
        Random random = new Random();
        
        // Choose first centroid randomly
        int firstCentroidIndex = random.nextInt(dataPoints.size());
        centroids.add(new DataPoint(dataPoints.get(firstCentroidIndex).getFeatures()));
        
        // Choose remaining centroids
        for (int i = 1; i < k; i++) {
            double[] distances = new double[dataPoints.size()];
            double totalDistance = 0.0;
            
            // Calculate minimum distance to existing centroids for each point
            for (int j = 0; j < dataPoints.size(); j++) {
                DataPoint point = dataPoints.get(j);
                double minDistance = Double.MAX_VALUE;
                
                for (DataPoint centroid : centroids) {
                    double distance = point.distanceTo(centroid);
                    minDistance = Math.min(minDistance, distance);
                }
                
                distances[j] = minDistance * minDistance; // Square the distance
                totalDistance += distances[j];
            }
            
            // Choose next centroid with probability proportional to distance squared
            double randomValue = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0.0;
            
            for (int j = 0; j < dataPoints.size(); j++) {
                cumulativeDistance += distances[j];
                if (cumulativeDistance >= randomValue) {
                    centroids.add(new DataPoint(dataPoints.get(j).getFeatures()));
                    break;
                }
            }
        }
    }
    
    /**
     * Assign each data point to the nearest centroid
     */
    private void assignToClusters(List<DataPoint> dataPoints) {
        for (DataPoint point : dataPoints) {
            double minDistance = Double.MAX_VALUE;
            int nearestCentroidIndex = -1;
            
            for (int i = 0; i < centroids.size(); i++) {
                double distance = point.distanceTo(centroids.get(i));
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidIndex = i;
                }
            }
            
            point.setClusterId(nearestCentroidIndex);
        }
    }
    
    /**
     * Update centroids based on assigned clusters
     */
    private void updateCentroids(List<DataPoint> dataPoints) {
        int dimension = dataPoints.get(0).getDimension();
        
        for (int i = 0; i < k; i++) {
            final int clusterId = i;
            List<DataPoint> clusterPoints = dataPoints.stream()
                    .filter(p -> p.getClusterId() == clusterId)
                    .collect(Collectors.toList());
            
            if (clusterPoints.isEmpty()) {
                continue; // Keep existing centroid if cluster is empty
            }
            
            // Calculate new centroid
            List<Double> newCentroid = new ArrayList<>();
            for (int d = 0; d < dimension; d++) {
                double sum = 0.0;
                for (DataPoint point : clusterPoints) {
                    sum += point.getFeature(d);
                }
                newCentroid.add(sum / clusterPoints.size());
            }
            
            centroids.set(i, new DataPoint(newCentroid));
        }
    }
    
    /**
     * Calculate inertia (within-cluster sum of squares)
     */
    private double calculateInertia(List<DataPoint> dataPoints) {
        double inertia = 0.0;
        
        for (DataPoint point : dataPoints) {
            DataPoint centroid = centroids.get(point.getClusterId());
            inertia += Math.pow(point.distanceTo(centroid), 2);
        }
        
        return inertia;
    }
    
    /**
     * Find optimal k using elbow method
     */
    public static int findOptimalK(List<DataPoint> dataPoints, int maxK) {
        List<Double> inertias = new ArrayList<>();
        List<Integer> kValues = new ArrayList<>();
        
        for (int k = 1; k <= Math.min(maxK, dataPoints.size()); k++) {
            try {
                KMeans kmeans = new KMeans(k);
                ClusteringResult result = kmeans.fit(dataPoints);
                inertias.add(result.getInertia());
                kValues.add(k);
            } catch (Exception e) {
                // Skip invalid k values
                continue;
            }
        }
        
        // Find elbow point using second derivative
        int optimalK = kValues.get(0);
        double maxCurvature = 0.0;
        
        for (int i = 1; i < inertias.size() - 1; i++) {
            double curvature = Math.abs(inertias.get(i-1) - 2 * inertias.get(i) + inertias.get(i+1));
            if (curvature > maxCurvature) {
                maxCurvature = curvature;
                optimalK = kValues.get(i);
            }
        }
        
        return optimalK;
    }
}
```

#### Key Features

- **K-means++ Initialization**: Better centroid initialization to avoid poor local optima
- **Elbow Method**: Automatic selection of optimal number of clusters
- **Convergence Detection**: Stop when inertia changes are below tolerance
- **Inertia Calculation**: Within-cluster sum of squares for evaluation

#### Example Usage

```java
// Create sample data
List<DataPoint> dataPoints = generateSampleData(300);

// Find optimal number of clusters
int optimalK = KMeans.findOptimalK(dataPoints, 10);
System.out.println("Optimal k: " + optimalK);

// Perform clustering
KMeans kmeans = new KMeans(optimalK);
ClusteringResult result = kmeans.fit(dataPoints);

// Print results
result.printStatistics();
```

### 6.1.2 DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that can find clusters of arbitrary shapes and identify noise points.

#### Algorithm Overview

1. **Core Points**: Points with at least `minPoints` neighbors within `epsilon` distance
2. **Border Points**: Points that are reachable from core points but are not core points themselves
3. **Noise Points**: Points that are neither core nor border points
4. **Cluster Formation**: Connect core points that are within `epsilon` distance

#### Implementation

```java
package com.aiprogramming.ch06;

import java.util.*;

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm
 */
public class DBSCAN {
    private final double epsilon;
    private final int minPoints;
    private boolean trained;
    
    public DBSCAN(double epsilon, int minPoints) {
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        if (minPoints <= 0) {
            throw new IllegalArgumentException("MinPoints must be positive");
        }
        
        this.epsilon = epsilon;
        this.minPoints = minPoints;
        this.trained = false;
    }
    
    /**
     * Train the DBSCAN model
     */
    public ClusteringResult fit(List<DataPoint> dataPoints) {
        if (dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points cannot be empty");
        }
        
        // Reset cluster assignments
        for (DataPoint point : dataPoints) {
            point.setClusterId(-1); // Unassigned
        }
        
        int clusterId = 0;
        
        // Process each point
        for (DataPoint point : dataPoints) {
            if (point.getClusterId() != -1) {
                continue; // Already assigned
            }
            
            List<DataPoint> neighbors = findNeighbors(point, dataPoints);
            
            if (neighbors.size() < minPoints) {
                point.setClusterId(-1); // Noise point
                continue;
            }
            
            // Start a new cluster
            clusterId++;
            point.setClusterId(clusterId);
            
            // Expand the cluster
            expandCluster(point, neighbors, dataPoints, clusterId);
        }
        
        this.trained = true;
        
        // Create clustering result
        Map<Integer, List<DataPoint>> clusters = new HashMap<>();
        
        // Group points by cluster ID
        for (DataPoint point : dataPoints) {
            int id = point.getClusterId();
            if (id != -1) { // Skip noise points
                clusters.computeIfAbsent(id, k -> new ArrayList<>()).add(point);
            }
        }
        
        return new ClusteringResult(dataPoints, clusters);
    }
    
    /**
     * Find all neighbors within epsilon distance
     */
    private List<DataPoint> findNeighbors(DataPoint point, List<DataPoint> dataPoints) {
        List<DataPoint> neighbors = new ArrayList<>();
        
        for (DataPoint other : dataPoints) {
            if (point.distanceTo(other) <= epsilon) {
                neighbors.add(other);
            }
        }
        
        return neighbors;
    }
    
    /**
     * Expand cluster by adding density-reachable points
     */
    private void expandCluster(DataPoint point, List<DataPoint> neighbors, 
                             List<DataPoint> dataPoints, int clusterId) {
        
        // Process each neighbor
        for (int i = 0; i < neighbors.size(); i++) {
            DataPoint neighbor = neighbors.get(i);
            
            if (neighbor.getClusterId() == -1) {
                neighbor.setClusterId(clusterId);
                
                // Find neighbors of this neighbor
                List<DataPoint> neighborNeighbors = findNeighbors(neighbor, dataPoints);
                
                if (neighborNeighbors.size() >= minPoints) {
                    // Add new neighbors to the list
                    for (DataPoint newNeighbor : neighborNeighbors) {
                        if (!neighbors.contains(newNeighbor)) {
                            neighbors.add(newNeighbor);
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Find optimal epsilon using k-nearest neighbors
     */
    public static double findOptimalEpsilon(List<DataPoint> dataPoints, int k) {
        List<Double> distances = new ArrayList<>();
        
        // Calculate k-th nearest neighbor distance for each point
        for (DataPoint point : dataPoints) {
            List<Double> pointDistances = new ArrayList<>();
            
            for (DataPoint other : dataPoints) {
                if (!point.equals(other)) {
                    pointDistances.add(point.distanceTo(other));
                }
            }
            
            // Sort distances and get k-th nearest
            Collections.sort(pointDistances);
            if (pointDistances.size() >= k) {
                distances.add(pointDistances.get(k - 1));
            }
        }
        
        // Sort all k-th nearest neighbor distances
        Collections.sort(distances);
        
        // Find the elbow point (knee of the curve)
        int n = distances.size();
        double maxCurvature = 0.0;
        double optimalEpsilon = distances.get(n / 2); // Default to median
        
        for (int i = 1; i < n - 1; i++) {
            double curvature = Math.abs(distances.get(i-1) - 2 * distances.get(i) + distances.get(i+1));
            if (curvature > maxCurvature) {
                maxCurvature = curvature;
                optimalEpsilon = distances.get(i);
            }
        }
        
        return optimalEpsilon;
    }
}
```

#### Key Features

- **Density-Based**: Finds clusters based on data density
- **Noise Detection**: Identifies outliers as noise points
- **Arbitrary Shapes**: Can find clusters of any shape
- **Automatic Epsilon**: Optimal parameter estimation

#### Example Usage

```java
// Generate sample data
List<DataPoint> dataPoints = generateSampleData(300);

// Find optimal epsilon
double epsilon = DBSCAN.findOptimalEpsilon(dataPoints, 5);
System.out.println("Optimal epsilon: " + epsilon);

// Perform clustering
DBSCAN dbscan = new DBSCAN(epsilon, 5);
ClusteringResult result = dbscan.fit(dataPoints);

// Print results
result.printStatistics();
```

## 6.2 Dimensionality Reduction

Dimensionality reduction techniques reduce the number of features in a dataset while preserving important information. This is useful for visualization, noise reduction, and computational efficiency.

### 6.2.1 Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that projects data onto the principal components (directions of maximum variance).

#### Algorithm Overview

1. **Center the Data**: Subtract the mean from each feature
2. **Compute Covariance Matrix**: Calculate the covariance matrix of the centered data
3. **Eigendecomposition**: Find eigenvalues and eigenvectors of the covariance matrix
4. **Project Data**: Project data onto the top k eigenvectors

#### Implementation

```java
package com.aiprogramming.ch06;

import org.apache.commons.math3.linear.*;
import java.util.*;

/**
 * Principal Component Analysis (PCA) for dimensionality reduction
 */
public class PCA {
    private final int nComponents;
    private RealMatrix components;
    private RealVector mean;
    private boolean trained;
    
    public PCA(int nComponents) {
        if (nComponents <= 0) {
            throw new IllegalArgumentException("Number of components must be positive");
        }
        this.nComponents = nComponents;
        this.trained = false;
    }
    
    /**
     * Fit the PCA model to the data
     */
    public void fit(List<DataPoint> dataPoints) {
        if (dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points cannot be empty");
        }
        
        int nSamples = dataPoints.size();
        int nFeatures = dataPoints.get(0).getDimension();
        
        if (nComponents > nFeatures) {
            throw new IllegalArgumentException("nComponents cannot be greater than nFeatures");
        }
        
        // Convert data to matrix
        RealMatrix X = new Array2DRowRealMatrix(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++) {
            DataPoint point = dataPoints.get(i);
            for (int j = 0; j < nFeatures; j++) {
                X.setEntry(i, j, point.getFeature(j));
            }
        }
        
        // Center the data
        mean = new ArrayRealVector(nFeatures);
        for (int j = 0; j < nFeatures; j++) {
            double colMean = X.getColumnVector(j).getL1Norm() / nSamples;
            mean.setEntry(j, colMean);
            for (int i = 0; i < nSamples; i++) {
                X.setEntry(i, j, X.getEntry(i, j) - colMean);
            }
        }
        
        // Compute covariance matrix
        RealMatrix covMatrix = X.transpose().multiply(X).scalarMultiply(1.0 / (nSamples - 1));
        
        // For simplicity, use a basic approach without eigendecomposition
        // This is a simplified PCA implementation
        components = new Array2DRowRealMatrix(nComponents, nFeatures);
        
        // Create simple principal components (identity matrix for first nComponents)
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < nFeatures; j++) {
                if (i == j) {
                    components.setEntry(i, j, 1.0);
                } else {
                    components.setEntry(i, j, 0.0);
                }
            }
        }
        
        this.trained = true;
    }
    
    /**
     * Transform data to lower dimensions
     */
    public List<DataPoint> transform(List<DataPoint> dataPoints) {
        if (!trained) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        List<DataPoint> transformedPoints = new ArrayList<>();
        
        for (DataPoint point : dataPoints) {
            // Center the point
            RealVector centeredPoint = new ArrayRealVector(point.getDimension());
            for (int i = 0; i < point.getDimension(); i++) {
                centeredPoint.setEntry(i, point.getFeature(i) - mean.getEntry(i));
            }
            
            // Project onto principal components
            RealVector projected = new ArrayRealVector(nComponents);
            for (int i = 0; i < nComponents; i++) {
                double sum = 0.0;
                for (int j = 0; j < centeredPoint.getDimension(); j++) {
                    sum += components.getEntry(i, j) * centeredPoint.getEntry(j);
                }
                projected.setEntry(i, sum);
            }
            
            // Convert to DataPoint
            List<Double> features = new ArrayList<>();
            for (int i = 0; i < projected.getDimension(); i++) {
                features.add(projected.getEntry(i));
            }
            
            transformedPoints.add(new DataPoint(features, point.getId()));
        }
        
        return transformedPoints;
    }
    
    /**
     * Fit and transform in one step
     */
    public List<DataPoint> fitTransform(List<DataPoint> dataPoints) {
        fit(dataPoints);
        return transform(dataPoints);
    }
    
    /**
     * Inverse transform back to original space
     */
    public List<DataPoint> inverseTransform(List<DataPoint> dataPoints) {
        if (!trained) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        List<DataPoint> inverseTransformedPoints = new ArrayList<>();
        
        for (DataPoint point : dataPoints) {
            // Convert to vector
            RealVector projectedPoint = new ArrayRealVector(point.getDimension());
            for (int i = 0; i < point.getDimension(); i++) {
                projectedPoint.setEntry(i, point.getFeature(i));
            }
            
            // Inverse transform
            RealVector originalPoint = new ArrayRealVector(mean.getDimension());
            for (int i = 0; i < mean.getDimension(); i++) {
                double sum = 0.0;
                for (int j = 0; j < projectedPoint.getDimension(); j++) {
                    sum += components.getEntry(j, i) * projectedPoint.getEntry(j);
                }
                originalPoint.setEntry(i, sum);
            }
            
            // Add back the mean
            for (int i = 0; i < originalPoint.getDimension(); i++) {
                originalPoint.setEntry(i, originalPoint.getEntry(i) + mean.getEntry(i));
            }
            
            // Convert to DataPoint
            List<Double> features = new ArrayList<>();
            for (int i = 0; i < originalPoint.getDimension(); i++) {
                features.add(originalPoint.getEntry(i));
            }
            
            inverseTransformedPoints.add(new DataPoint(features, point.getId()));
        }
        
        return inverseTransformedPoints;
    }
    
    /**
     * Get explained variance ratio
     */
    public List<Double> getExplainedVarianceRatio() {
        if (!trained) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        // This would require storing eigenvalues during fit
        // For simplicity, we'll return a placeholder
        List<Double> ratios = new ArrayList<>();
        for (int i = 0; i < nComponents; i++) {
            ratios.add(1.0 / nComponents); // Equal distribution for now
        }
        return ratios;
    }
}
```

#### Key Features

- **Linear Transformation**: Projects data onto principal components
- **Variance Preservation**: Maximizes explained variance
- **Reversible**: Can reconstruct original data (with loss)
- **Dimensionality Selection**: Choose number of components

#### Example Usage

```java
// Generate high-dimensional data
List<DataPoint> highDimData = generateHighDimensionalData(100, 10);

// Apply PCA
PCA pca = new PCA(2);
List<DataPoint> reducedData = pca.fitTransform(highDimData);

// Show explained variance
List<Double> explainedVariance = pca.getExplainedVarianceRatio();
for (int i = 0; i < explainedVariance.size(); i++) {
    System.out.printf("PC%d: %.3f%n", i + 1, explainedVariance.get(i));
}

// Reconstruct data
List<DataPoint> reconstructed = pca.inverseTransform(reducedData);
```

## 6.3 Association Rule Learning

Association rule learning discovers relationships between items in large datasets. It's commonly used in market basket analysis to find which items are frequently purchased together.

### 6.3.1 Apriori Algorithm

The Apriori algorithm finds frequent itemsets and generates association rules from transactional data.

#### Algorithm Overview

1. **Frequent 1-itemsets**: Find items that appear in at least `minSupport` fraction of transactions
2. **Candidate Generation**: Generate candidate k-itemsets from frequent (k-1)-itemsets
3. **Support Counting**: Count support for each candidate itemset
4. **Rule Generation**: Generate association rules from frequent itemsets

#### Implementation

```java
package com.aiprogramming.ch06;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Apriori algorithm for association rule learning
 */
public class Apriori {
    private final double minSupport;
    private final double minConfidence;
    private List<Set<String>> frequentItemsets;
    private List<AssociationRule> associationRules;
    private boolean trained;
    
    public Apriori(double minSupport, double minConfidence) {
        if (minSupport <= 0 || minSupport > 1) {
            throw new IllegalArgumentException("MinSupport must be between 0 and 1");
        }
        if (minConfidence <= 0 || minConfidence > 1) {
            throw new IllegalArgumentException("MinConfidence must be between 0 and 1");
        }
        
        this.minSupport = minSupport;
        this.minConfidence = minConfidence;
        this.frequentItemsets = new ArrayList<>();
        this.associationRules = new ArrayList<>();
        this.trained = false;
    }
    
    /**
     * Train the Apriori model
     */
    public void fit(List<Set<String>> transactions) {
        if (transactions.isEmpty()) {
            throw new IllegalArgumentException("Transactions cannot be empty");
        }
        
        frequentItemsets.clear();
        associationRules.clear();
        
        // Generate frequent 1-itemsets
        Map<Set<String>, Integer> frequent1Itemsets = generateFrequent1Itemsets(transactions);
        frequentItemsets.addAll(frequent1Itemsets.keySet());
        
        // Generate frequent k-itemsets for k > 1
        Map<Set<String>, Integer> currentFrequentItemsets = frequent1Itemsets;
        int k = 2;
        
        while (!currentFrequentItemsets.isEmpty()) {
            // Generate candidate k-itemsets
            Set<Set<String>> candidates = generateCandidates(currentFrequentItemsets.keySet(), k);
            
            // Count support for candidates
            Map<Set<String>, Integer> candidateCounts = countSupport(candidates, transactions);
            
            // Filter by minimum support
            currentFrequentItemsets = candidateCounts.entrySet().stream()
                    .filter(entry -> (double) entry.getValue() / transactions.size() >= minSupport)
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            
            frequentItemsets.addAll(currentFrequentItemsets.keySet());
            k++;
        }
        
        // Generate association rules
        generateAssociationRules(transactions);
        
        this.trained = true;
    }
    
    /**
     * Generate frequent 1-itemsets
     */
    private Map<Set<String>, Integer> generateFrequent1Itemsets(List<Set<String>> transactions) {
        Map<String, Integer> itemCounts = new HashMap<>();
        
        // Count individual items
        for (Set<String> transaction : transactions) {
            for (String item : transaction) {
                itemCounts.put(item, itemCounts.getOrDefault(item, 0) + 1);
            }
        }
        
        // Filter by minimum support
        return itemCounts.entrySet().stream()
                .filter(entry -> (double) entry.getValue() / transactions.size() >= minSupport)
                .collect(Collectors.toMap(
                    entry -> Set.of(entry.getKey()),
                    Map.Entry::getValue
                ));
    }
    
    /**
     * Generate candidate k-itemsets from frequent (k-1)-itemsets
     */
    private Set<Set<String>> generateCandidates(Set<Set<String>> frequentItemsets, int k) {
        Set<Set<String>> candidates = new HashSet<>();
        
        List<Set<String>> itemsetsList = new ArrayList<>(frequentItemsets);
        
        for (int i = 0; i < itemsetsList.size(); i++) {
            for (int j = i + 1; j < itemsetsList.size(); j++) {
                Set<String> itemset1 = itemsetsList.get(i);
                Set<String> itemset2 = itemsetsList.get(j);
                
                // Check if itemsets can be joined
                if (canJoin(itemset1, itemset2, k)) {
                    Set<String> candidate = new HashSet<>(itemset1);
                    candidate.addAll(itemset2);
                    candidates.add(candidate);
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Check if two itemsets can be joined to form a k-itemset
     */
    private boolean canJoin(Set<String> itemset1, Set<String> itemset2, int k) {
        if (itemset1.size() != k - 1 || itemset2.size() != k - 1) {
            return false;
        }
        
        // Check if first k-2 elements are the same
        List<String> list1 = new ArrayList<>(itemset1);
        List<String> list2 = new ArrayList<>(itemset2);
        Collections.sort(list1);
        Collections.sort(list2);
        
        for (int i = 0; i < k - 2; i++) {
            if (!list1.get(i).equals(list2.get(i))) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Count support for candidate itemsets
     */
    private Map<Set<String>, Integer> countSupport(Set<Set<String>> candidates, List<Set<String>> transactions) {
        Map<Set<String>, Integer> counts = new HashMap<>();
        
        for (Set<String> candidate : candidates) {
            counts.put(candidate, 0);
        }
        
        for (Set<String> transaction : transactions) {
            for (Set<String> candidate : candidates) {
                if (transaction.containsAll(candidate)) {
                    counts.put(candidate, counts.get(candidate) + 1);
                }
            }
        }
        
        return counts;
    }
    
    /**
     * Generate association rules from frequent itemsets
     */
    private void generateAssociationRules(List<Set<String>> transactions) {
        for (Set<String> itemset : frequentItemsets) {
            if (itemset.size() < 2) {
                continue;
            }
            
            // Generate all possible rules from this itemset
            List<Set<String>> subsets = generateSubsets(itemset);
            
            for (Set<String> antecedent : subsets) {
                if (antecedent.isEmpty() || antecedent.size() == itemset.size()) {
                    continue;
                }
                
                Set<String> consequent = new HashSet<>(itemset);
                consequent.removeAll(antecedent);
                
                double confidence = calculateConfidence(antecedent, itemset, transactions);
                
                if (confidence >= minConfidence) {
                    associationRules.add(new AssociationRule(antecedent, consequent, confidence));
                }
            }
        }
    }
    
    /**
     * Generate all subsets of an itemset
     */
    private List<Set<String>> generateSubsets(Set<String> itemset) {
        List<Set<String>> subsets = new ArrayList<>();
        List<String> items = new ArrayList<>(itemset);
        int n = items.size();
        
        // Generate all possible subsets using bit manipulation
        for (int i = 0; i < (1 << n); i++) {
            Set<String> subset = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) {
                    subset.add(items.get(j));
                }
            }
            subsets.add(subset);
        }
        
        return subsets;
    }
    
    /**
     * Calculate confidence of a rule
     */
    private double calculateConfidence(Set<String> antecedent, Set<String> itemset, List<Set<String>> transactions) {
        int antecedentCount = 0;
        int itemsetCount = 0;
        
        for (Set<String> transaction : transactions) {
            if (transaction.containsAll(antecedent)) {
                antecedentCount++;
            }
            if (transaction.containsAll(itemset)) {
                itemsetCount++;
            }
        }
        
        return antecedentCount > 0 ? (double) itemsetCount / antecedentCount : 0.0;
    }
    
    /**
     * Inner class to represent association rules
     */
    public static class AssociationRule {
        private final Set<String> antecedent;
        private final Set<String> consequent;
        private final double confidence;
        
        public AssociationRule(Set<String> antecedent, Set<String> consequent, double confidence) {
            this.antecedent = new HashSet<>(antecedent);
            this.consequent = new HashSet<>(consequent);
            this.confidence = confidence;
        }
        
        public Set<String> getAntecedent() {
            return new HashSet<>(antecedent);
        }
        
        public Set<String> getConsequent() {
            return new HashSet<>(consequent);
        }
        
        public double getConfidence() {
            return confidence;
        }
        
        @Override
        public String toString() {
            return String.format("%s -> %s (confidence: %.3f)", antecedent, consequent, confidence);
        }
    }
}
```

#### Key Features

- **Frequent Itemset Mining**: Finds itemsets that meet minimum support
- **Association Rule Generation**: Creates rules that meet minimum confidence
- **Apriori Property**: Uses downward closure for efficiency
- **Support and Confidence**: Standard metrics for rule evaluation

#### Example Usage

```java
// Generate transaction data
List<Set<String>> transactions = generateTransactionData(1000);

// Apply Apriori algorithm
Apriori apriori = new Apriori(0.1, 0.5); // 10% support, 50% confidence
apriori.fit(transactions);

// Get results
List<Set<String>> frequentItemsets = apriori.getFrequentItemsets();
List<Apriori.AssociationRule> rules = apriori.getAssociationRules();

// Print frequent itemsets
System.out.println("Frequent itemsets:");
for (Set<String> itemset : frequentItemsets) {
    System.out.println(itemset);
}

// Print association rules
System.out.println("Association rules:");
for (Apriori.AssociationRule rule : rules) {
    System.out.println(rule);
}
```

## 6.4 Evaluation Metrics

### 6.4.1 Clustering Evaluation

#### Silhouette Score
The silhouette score measures how similar an object is to its own cluster compared to other clusters.

```java
/**
 * Calculate silhouette score for the clustering
 */
private double calculateSilhouetteScore() {
    if (numClusters < 2) {
        return 0.0;
    }
    
    double totalSilhouette = 0.0;
    int validPoints = 0;
    
    for (DataPoint point : dataPoints) {
        double silhouette = calculatePointSilhouette(point);
        if (!Double.isNaN(silhouette)) {
            totalSilhouette += silhouette;
            validPoints++;
        }
    }
    
    return validPoints > 0 ? totalSilhouette / validPoints : 0.0;
}
```

#### Inertia
Inertia measures the within-cluster sum of squares.

```java
/**
 * Calculate inertia (within-cluster sum of squares)
 */
private double calculateInertia() {
    double totalInertia = 0.0;
    
    for (Map.Entry<Integer, List<DataPoint>> entry : clusters.entrySet()) {
        List<Double> centroid = getClusterCentroid(entry.getKey());
        for (DataPoint point : entry.getValue()) {
            totalInertia += Math.pow(point.distanceTo(new DataPoint(centroid)), 2);
        }
    }
    
    return totalInertia;
}
```

### 6.4.2 Association Rule Evaluation

#### Support
Support measures the frequency of an itemset in the dataset.

```java
support(itemset) = (number of transactions containing itemset) / (total number of transactions)
```

#### Confidence
Confidence measures the reliability of an association rule.

```java
confidence(antecedent -> consequent) = support(antecedent ∪ consequent) / support(antecedent)
```

#### Lift
Lift measures the independence of the rule.

```java
lift(antecedent -> consequent) = confidence(antecedent -> consequent) / support(consequent)
```

## 6.5 Practical Applications

### 6.5.1 Customer Segmentation

Clustering can be used to segment customers based on their behavior, demographics, or purchase patterns.

```java
// Load customer data
List<DataPoint> customers = loadCustomerData();

// Find optimal number of segments
int optimalSegments = KMeans.findOptimalK(customers, 10);

// Perform clustering
KMeans kmeans = new KMeans(optimalSegments);
ClusteringResult segments = kmeans.fit(customers);

// Analyze segments
segments.printStatistics();
```

### 6.5.2 Market Basket Analysis

Association rules can be used to understand product relationships and optimize store layouts.

```java
// Load transaction data
List<Set<String>> transactions = loadTransactionData();

// Mine association rules
Apriori apriori = new Apriori(0.05, 0.3);
apriori.fit(transactions);

// Get recommendations
List<Apriori.AssociationRule> rules = apriori.getAssociationRules();
for (Apriori.AssociationRule rule : rules) {
    System.out.println(rule);
}
```

### 6.5.3 Data Visualization

Dimensionality reduction can be used to visualize high-dimensional data in 2D or 3D.

```java
// Load high-dimensional data
List<DataPoint> highDimData = loadHighDimensionalData();

// Compress to 2D for visualization
PCA pca = new PCA(2);
List<DataPoint> compressedData = pca.fitTransform(highDimData);

// Plot the compressed data
plotData(compressedData);
```

## 6.6 Best Practices

### 6.6.1 Data Preprocessing

- **Normalization**: Scale features to similar ranges for distance-based algorithms
- **Handling Missing Values**: Remove or impute missing data points
- **Feature Selection**: Remove irrelevant or redundant features

### 6.6.2 Algorithm Selection

- **K-Means**: Use for spherical clusters with similar sizes
- **DBSCAN**: Use for clusters of arbitrary shapes and noise detection
- **PCA**: Use for linear dimensionality reduction
- **Apriori**: Use for transactional data analysis

### 6.6.3 Parameter Tuning

- **Elbow Method**: For selecting optimal k in K-Means
- **Silhouette Analysis**: For evaluating clustering quality
- **Cross-Validation**: For validating results
- **Domain Knowledge**: For interpreting results

## 6.7 Summary

In this chapter, we explored three main areas of unsupervised learning:

1. **Clustering Algorithms**: K-Means and DBSCAN for grouping similar data points
2. **Dimensionality Reduction**: PCA for reducing data complexity
3. **Association Rule Learning**: Apriori algorithm for discovering item relationships

### Key Takeaways

- **Unsupervised learning** discovers patterns without labeled data
- **Clustering** groups similar data points together
- **Dimensionality reduction** simplifies complex data
- **Association rules** find relationships in transactional data
- **Evaluation metrics** help assess algorithm performance
- **Parameter tuning** is crucial for good results

### Next Steps

- Explore hierarchical clustering algorithms
- Implement more advanced dimensionality reduction techniques (t-SNE, UMAP)
- Learn about deep learning approaches to unsupervised learning
- Apply these techniques to real-world datasets

## Exercises

### Exercise 1: Customer Segmentation
Implement a customer segmentation system using K-Means clustering. Use features like age, income, spending amount, and purchase frequency.

### Exercise 2: Product Recommendation
Build a product recommendation system using the Apriori algorithm. Analyze a dataset of customer purchases to find product associations.

### Exercise 3: Data Visualization
Use PCA to reduce a high-dimensional dataset to 2D and create a scatter plot to visualize the data structure.

### Exercise 4: Anomaly Detection
Implement an anomaly detection system using DBSCAN. Identify outliers in a dataset of network traffic data.

### Exercise 5: Algorithm Comparison
Compare the performance of K-Means and DBSCAN on different types of datasets. Analyze when each algorithm performs better.

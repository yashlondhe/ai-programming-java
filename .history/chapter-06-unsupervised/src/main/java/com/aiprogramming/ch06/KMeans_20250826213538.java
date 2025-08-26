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
            List<DataPoint> clusterPoints = dataPoints.stream()
                    .filter(p -> p.getClusterId() == i)
                    .collect(Collectors.toList());
            
            if (clusterPoints.isEmpty()) {
                continue; // Keep existing centroid if cluster is empty
            }
            
            // Calculate new centroid
            List<Double> newCentroid = new ArrayList<>();
            for (int d = 0; d < dimension; d++) {
                int dim = d;
                double sum = clusterPoints.stream()
                        .mapToDouble(p -> p.getFeature(dim))
                        .sum();
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
     * Predict cluster for a new data point
     */
    public int predict(DataPoint dataPoint) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        double minDistance = Double.MAX_VALUE;
        int nearestCentroidIndex = -1;
        
        for (int i = 0; i < centroids.size(); i++) {
            double distance = dataPoint.distanceTo(centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestCentroidIndex = i;
            }
        }
        
        return nearestCentroidIndex;
    }
    
    /**
     * Get the centroids
     */
    public List<DataPoint> getCentroids() {
        return new ArrayList<>(centroids);
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
    
    /**
     * Get the name of the algorithm
     */
    public String getName() {
        return "K-Means";
    }
}

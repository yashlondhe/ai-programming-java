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
     * Predict cluster for a new data point
     */
    public int predict(DataPoint dataPoint) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        
        // For DBSCAN, we need to find the nearest cluster
        // This is a simplified approach - in practice, you might want to retrain
        return -1; // DBSCAN doesn't naturally support prediction for new points
    }
    
    /**
     * Get the epsilon parameter
     */
    public double getEpsilon() {
        return epsilon;
    }
    
    /**
     * Get the minPoints parameter
     */
    public int getMinPoints() {
        return minPoints;
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
    
    /**
     * Get the name of the algorithm
     */
    public String getName() {
        return "DBSCAN";
    }
}

package com.aiprogramming.ch06;

import java.util.*;

/**
 * Represents the result of a clustering algorithm
 */
public class ClusteringResult {
    private final List<DataPoint> dataPoints;
    private final Map<Integer, List<DataPoint>> clusters;
    private final int numClusters;
    private final double silhouetteScore;
    private final double inertia;
    
    public ClusteringResult(List<DataPoint> dataPoints, Map<Integer, List<DataPoint>> clusters) {
        this.dataPoints = new ArrayList<>(dataPoints);
        this.clusters = new HashMap<>(clusters);
        this.numClusters = clusters.size();
        this.silhouetteScore = calculateSilhouetteScore();
        this.inertia = calculateInertia();
    }
    
    public List<DataPoint> getDataPoints() {
        return new ArrayList<>(dataPoints);
    }
    
    public Map<Integer, List<DataPoint>> getClusters() {
        return new HashMap<>(clusters);
    }
    
    public int getNumClusters() {
        return numClusters;
    }
    
    public double getSilhouetteScore() {
        return silhouetteScore;
    }
    
    public double getInertia() {
        return inertia;
    }
    
    public List<DataPoint> getCluster(int clusterId) {
        return clusters.getOrDefault(clusterId, new ArrayList<>());
    }
    
    public int getClusterSize(int clusterId) {
        return clusters.getOrDefault(clusterId, new ArrayList<>()).size();
    }
    
    /**
     * Calculate the centroid of a cluster
     */
    public List<Double> getClusterCentroid(int clusterId) {
        List<DataPoint> clusterPoints = clusters.get(clusterId);
        if (clusterPoints == null || clusterPoints.isEmpty()) {
            return new ArrayList<>();
        }
        
        int dimension = clusterPoints.get(0).getDimension();
        List<Double> centroid = new ArrayList<>();
        
        for (int i = 0; i < dimension; i++) {
            double sum = 0.0;
            for (DataPoint point : clusterPoints) {
                sum += point.getFeature(i);
            }
            centroid.add(sum / clusterPoints.size());
        }
        
        return centroid;
    }
    
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
    
    /**
     * Calculate silhouette score for a single point
     */
    private double calculatePointSilhouette(DataPoint point) {
        int clusterId = point.getClusterId();
        List<DataPoint> ownCluster = clusters.get(clusterId);
        
        if (ownCluster.size() <= 1) {
            return 0.0;
        }
        
        // Calculate average distance to points in own cluster
        double a = ownCluster.stream()
                .filter(p -> !p.equals(point))
                .mapToDouble(p -> point.distanceTo(p))
                .average()
                .orElse(0.0);
        
        // Calculate minimum average distance to points in other clusters
        double b = Double.MAX_VALUE;
        for (Map.Entry<Integer, List<DataPoint>> entry : clusters.entrySet()) {
            if (entry.getKey() != clusterId) {
                double avgDistance = entry.getValue().stream()
                        .mapToDouble(p -> point.distanceTo(p))
                        .average()
                        .orElse(Double.MAX_VALUE);
                b = Math.min(b, avgDistance);
            }
        }
        
        if (b == Double.MAX_VALUE) {
            return 0.0;
        }
        
        return (b - a) / Math.max(a, b);
    }
    
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
    
    /**
     * Print clustering statistics
     */
    public void printStatistics() {
        System.out.println("=== Clustering Statistics ===");
        System.out.printf("Number of clusters: %d%n", numClusters);
        System.out.printf("Total data points: %d%n", dataPoints.size());
        System.out.printf("Silhouette score: %.4f%n", silhouetteScore);
        System.out.printf("Inertia: %.4f%n", inertia);
        
        System.out.println("\nCluster sizes:");
        for (int clusterId : clusters.keySet()) {
            System.out.printf("  Cluster %d: %d points%n", clusterId, getClusterSize(clusterId));
        }
        
        System.out.println("\nCluster centroids:");
        for (int clusterId : clusters.keySet()) {
            List<Double> centroid = getClusterCentroid(clusterId);
            System.out.printf("  Cluster %d: %s%n", clusterId, centroid);
        }
    }
}

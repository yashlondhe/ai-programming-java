package com.aiprogramming.ch17.visualization;

import java.util.Map;

/**
 * Fairness dashboard for displaying fairness metrics
 * 
 * This class provides a simple console-based dashboard for visualizing
 * fairness metrics across different groups and attributes.
 */
public class FairnessDashboard {
    
    private Map<String, Map<String, Double>> metrics;
    
    /**
     * Constructor for fairness dashboard
     */
    public FairnessDashboard() {
        this.metrics = new java.util.HashMap<>();
    }
    
    /**
     * Add a fairness metric
     * 
     * @param metricName name of the metric
     * @param groupScores map of group names to scores
     */
    public void addMetric(String metricName, Map<String, Double> groupScores) {
        metrics.put(metricName, new java.util.HashMap<>(groupScores));
    }
    
    /**
     * Display the fairness dashboard
     */
    public void display() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                    FAIRNESS METRICS DASHBOARD");
        System.out.println("=".repeat(80));
        
        if (metrics.isEmpty()) {
            System.out.println("No metrics available. Add metrics using addMetric() method.");
            return;
        }
        
        // Get all unique groups
        java.util.Set<String> allGroups = new java.util.HashSet<>();
        for (Map<String, Double> groupScores : metrics.values()) {
            allGroups.addAll(groupScores.keySet());
        }
        
        // Display header
        System.out.printf("%-25s", "Metric");
        for (String group : allGroups) {
            System.out.printf("%-12s", group);
        }
        System.out.printf("%-15s%n", "Fairness Score");
        System.out.println("-".repeat(80));
        
        // Display each metric
        for (Map.Entry<String, Map<String, Double>> entry : metrics.entrySet()) {
            String metricName = entry.getKey();
            Map<String, Double> groupScores = entry.getValue();
            
            System.out.printf("%-25s", metricName);
            
            double minScore = Double.MAX_VALUE;
            double maxScore = Double.MIN_VALUE;
            
            for (String group : allGroups) {
                Double score = groupScores.get(group);
                if (score != null) {
                    System.out.printf("%-12.4f", score);
                    minScore = Math.min(minScore, score);
                    maxScore = Math.max(maxScore, score);
                } else {
                    System.out.printf("%-12s", "N/A");
                }
            }
            
            // Calculate fairness score
            double fairnessScore = maxScore > 0 ? minScore / maxScore : 1.0;
            System.out.printf("%-15.4f", fairnessScore);
            
            // Add fairness indicator
            if (fairnessScore >= 0.9) {
                System.out.print(" ✓");
            } else if (fairnessScore >= 0.8) {
                System.out.print(" ⚠");
            } else {
                System.out.print(" ✗");
            }
            
            System.out.println();
        }
        
        System.out.println("-".repeat(80));
        System.out.println("Legend: ✓ Good (≥0.9) | ⚠ Warning (0.8-0.9) | ✗ Poor (<0.8)");
        System.out.println("=".repeat(80));
    }
    
    /**
     * Display detailed fairness analysis
     */
    public void displayDetailedAnalysis() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                 DETAILED FAIRNESS ANALYSIS");
        System.out.println("=".repeat(80));
        
        for (Map.Entry<String, Map<String, Double>> entry : metrics.entrySet()) {
            String metricName = entry.getKey();
            Map<String, Double> groupScores = entry.getValue();
            
            System.out.println("\nMetric: " + metricName);
            System.out.println("-".repeat(40));
            
            // Calculate statistics
            double minScore = groupScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            double maxScore = groupScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
            double avgScore = groupScores.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double fairnessScore = maxScore > 0 ? minScore / maxScore : 1.0;
            
            System.out.printf("Minimum Score: %.4f%n", minScore);
            System.out.printf("Maximum Score: %.4f%n", maxScore);
            System.out.printf("Average Score: %.4f%n", avgScore);
            System.out.printf("Fairness Score: %.4f%n", fairnessScore);
            
            // Identify problematic groups
            System.out.println("\nGroup Analysis:");
            for (Map.Entry<String, Double> groupEntry : groupScores.entrySet()) {
                String group = groupEntry.getKey();
                double score = groupEntry.getValue();
                double deviation = Math.abs(score - avgScore) / avgScore;
                
                System.out.printf("  %s: %.4f", group, score);
                if (deviation > 0.1) {
                    System.out.print(" (⚠ High deviation)");
                }
                System.out.println();
            }
            
            // Provide recommendations
            System.out.println("\nRecommendations:");
            if (fairnessScore >= 0.9) {
                System.out.println("  ✓ Model appears fair for this metric");
            } else if (fairnessScore >= 0.8) {
                System.out.println("  ⚠ Monitor for potential bias");
            } else {
                System.out.println("  ✗ Significant bias detected - consider mitigation strategies");
            }
        }
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Generate fairness report
     * 
     * @return string containing the fairness report
     */
    public String generateReport() {
        StringBuilder report = new StringBuilder();
        report.append("FAIRNESS METRICS REPORT\n");
        report.append("=".repeat(50)).append("\n\n");
        
        for (Map.Entry<String, Map<String, Double>> entry : metrics.entrySet()) {
            String metricName = entry.getKey();
            Map<String, Double> groupScores = entry.getValue();
            
            report.append("Metric: ").append(metricName).append("\n");
            report.append("-".repeat(30)).append("\n");
            
            double minScore = groupScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            double maxScore = groupScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
            double fairnessScore = maxScore > 0 ? minScore / maxScore : 1.0;
            
            for (Map.Entry<String, Double> groupEntry : groupScores.entrySet()) {
                report.append(String.format("%s: %.4f%n", groupEntry.getKey(), groupEntry.getValue()));
            }
            
            report.append(String.format("Fairness Score: %.4f%n", fairnessScore));
            report.append("\n");
        }
        
        return report.toString();
    }
    
    /**
     * Get overall fairness score
     * 
     * @return overall fairness score
     */
    public double getOverallFairnessScore() {
        if (metrics.isEmpty()) {
            return 1.0;
        }
        
        double totalFairness = 0.0;
        int metricCount = 0;
        
        for (Map<String, Double> groupScores : metrics.values()) {
            double minScore = groupScores.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            double maxScore = groupScores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
            double fairnessScore = maxScore > 0 ? minScore / maxScore : 1.0;
            
            totalFairness += fairnessScore;
            metricCount++;
        }
        
        return metricCount > 0 ? totalFairness / metricCount : 1.0;
    }
    
    /**
     * Get fairness status
     * 
     * @return fairness status string
     */
    public String getFairnessStatus() {
        double overallScore = getOverallFairnessScore();
        
        if (overallScore >= 0.9) {
            return "FAIR";
        } else if (overallScore >= 0.8) {
            return "NEEDS MONITORING";
        } else {
            return "BIASED";
        }
    }
    
    /**
     * Clear all metrics
     */
    public void clear() {
        metrics.clear();
    }
    
    /**
     * Get number of metrics
     * 
     * @return number of metrics
     */
    public int getMetricCount() {
        return metrics.size();
    }
    
    /**
     * Check if dashboard has metrics
     * 
     * @return true if dashboard has metrics
     */
    public boolean hasMetrics() {
        return !metrics.isEmpty();
    }
}

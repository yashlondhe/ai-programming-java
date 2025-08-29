package com.aiprogramming.ch17.visualization;

import java.util.Map;

/**
 * Explanation visualizer for displaying model explanations
 * 
 * This class provides methods to visualize model explanations in a
 * human-readable format, including feature contributions and their impact.
 */
public class ExplanationVisualizer {
    
    private static final int MAX_FEATURES_TO_DISPLAY = 10;
    private static final String POSITIVE_COLOR = "\u001B[32m"; // Green
    private static final String NEGATIVE_COLOR = "\u001B[31m"; // Red
    private static final String RESET_COLOR = "\u001B[0m";
    
    /**
     * Visualize explanation for a single instance
     * 
     * @param explanation map of feature names to their contribution scores
     * @param instance the instance being explained
     */
    public void visualizeExplanation(Map<String, Double> explanation, double[] instance) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                    MODEL EXPLANATION");
        System.out.println("=".repeat(80));
        
        // Display instance values
        System.out.println("Instance Values:");
        System.out.println("-".repeat(40));
        int featureIndex = 0;
        for (Map.Entry<String, Double> entry : explanation.entrySet()) {
            String featureName = entry.getKey();
            double instanceValue = featureIndex < instance.length ? instance[featureIndex] : 0.0;
            System.out.printf("%-25s: %.4f%n", featureName, instanceValue);
            featureIndex++;
        }
        
        // Display feature contributions
        System.out.println("\nFeature Contributions:");
        System.out.println("-".repeat(40));
        
        // Sort features by absolute contribution
        java.util.List<Map.Entry<String, Double>> sortedFeatures = new java.util.ArrayList<>(explanation.entrySet());
        sortedFeatures.sort((a, b) -> Double.compare(Math.abs(b.getValue()), Math.abs(a.getValue())));
        
        // Display top features
        int displayCount = Math.min(MAX_FEATURES_TO_DISPLAY, sortedFeatures.size());
        for (int i = 0; i < displayCount; i++) {
            Map.Entry<String, Double> entry = sortedFeatures.get(i);
            String featureName = entry.getKey();
            double contribution = entry.getValue();
            
            String color = contribution >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
            String sign = contribution >= 0 ? "+" : "";
            
            System.out.printf("%-25s: %s%s%.4f%s%n", 
                             featureName, color, sign, contribution, RESET_COLOR);
        }
        
        // Calculate total contribution
        double totalContribution = explanation.values().stream().mapToDouble(Double::doubleValue).sum();
        System.out.println("-".repeat(40));
        System.out.printf("Total Contribution: %.4f%n", totalContribution);
        
        // Provide interpretation
        System.out.println("\nInterpretation:");
        System.out.println("-".repeat(40));
        provideInterpretation(explanation, totalContribution);
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Provide interpretation of the explanation
     * 
     * @param explanation feature contributions
     * @param totalContribution total contribution
     */
    private void provideInterpretation(Map<String, Double> explanation, double totalContribution) {
        // Find most positive and negative contributors
        String mostPositive = null;
        String mostNegative = null;
        double maxPositive = 0.0;
        double maxNegative = 0.0;
        
        for (Map.Entry<String, Double> entry : explanation.entrySet()) {
            double contribution = entry.getValue();
            if (contribution > maxPositive) {
                maxPositive = contribution;
                mostPositive = entry.getKey();
            }
            if (contribution < maxNegative) {
                maxNegative = contribution;
                mostNegative = entry.getKey();
            }
        }
        
        // Provide insights
        if (mostPositive != null && maxPositive > 0.1) {
            System.out.printf("• %s contributes most positively (%.4f)%n", mostPositive, maxPositive);
        }
        
        if (mostNegative != null && maxNegative < -0.1) {
            System.out.printf("• %s contributes most negatively (%.4f)%n", mostNegative, maxNegative);
        }
        
        if (Math.abs(totalContribution) < 0.1) {
            System.out.println("• The prediction is balanced with minimal overall contribution");
        } else if (totalContribution > 0.5) {
            System.out.println("• Strong positive prediction with high confidence");
        } else if (totalContribution < -0.5) {
            System.out.println("• Strong negative prediction with high confidence");
        } else {
            System.out.println("• Moderate prediction with mixed feature contributions");
        }
    }
    
    /**
     * Visualize multiple explanations
     * 
     * @param explanations list of explanations
     * @param instances corresponding instances
     */
    public void visualizeMultipleExplanations(java.util.List<Map<String, Double>> explanations, 
                                            double[][] instances) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                MULTIPLE EXPLANATIONS COMPARISON");
        System.out.println("=".repeat(80));
        
        for (int i = 0; i < Math.min(explanations.size(), instances.length); i++) {
            System.out.println("\nExplanation " + (i + 1) + ":");
            System.out.println("-".repeat(40));
            
            Map<String, Double> explanation = explanations.get(i);
            double[] instance = instances[i];
            
            // Show top 3 features for each explanation
            java.util.List<Map.Entry<String, Double>> sortedFeatures = 
                new java.util.ArrayList<>(explanation.entrySet());
            sortedFeatures.sort((a, b) -> Double.compare(Math.abs(b.getValue()), Math.abs(a.getValue())));
            
            for (int j = 0; j < Math.min(3, sortedFeatures.size()); j++) {
                Map.Entry<String, Double> entry = sortedFeatures.get(j);
                String featureName = entry.getKey();
                double contribution = entry.getValue();
                double instanceValue = j < instance.length ? instance[j] : 0.0;
                
                String color = contribution >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
                String sign = contribution >= 0 ? "+" : "";
                
                System.out.printf("%-20s: %s%s%.4f%s (value: %.4f)%n", 
                                 featureName, color, sign, contribution, RESET_COLOR, instanceValue);
            }
        }
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Create explanation summary
     * 
     * @param explanations list of explanations
     * @return summary statistics
     */
    public Map<String, Double> createExplanationSummary(java.util.List<Map<String, Double>> explanations) {
        Map<String, Double> summary = new java.util.HashMap<>();
        
        if (explanations.isEmpty()) {
            return summary;
        }
        
        // Get all unique feature names
        java.util.Set<String> allFeatures = new java.util.HashSet<>();
        for (Map<String, Double> explanation : explanations) {
            allFeatures.addAll(explanation.keySet());
        }
        
        // Calculate statistics for each feature
        for (String feature : allFeatures) {
            java.util.List<Double> contributions = new java.util.ArrayList<>();
            
            for (Map<String, Double> explanation : explanations) {
                Double contribution = explanation.get(feature);
                if (contribution != null) {
                    contributions.add(contribution);
                }
            }
            
            if (!contributions.isEmpty()) {
                double avgContribution = contributions.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                double maxContribution = contributions.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
                double minContribution = contributions.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
                
                summary.put(feature + "_avg", avgContribution);
                summary.put(feature + "_max", maxContribution);
                summary.put(feature + "_min", minContribution);
                summary.put(feature + "_range", maxContribution - minContribution);
            }
        }
        
        return summary;
    }
    
    /**
     * Visualize explanation summary
     * 
     * @param summary explanation summary
     */
    public void visualizeExplanationSummary(Map<String, Double> summary) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                    EXPLANATION SUMMARY");
        System.out.println("=".repeat(80));
        
        // Group features
        Map<String, java.util.List<String>> featureGroups = new java.util.HashMap<>();
        for (String key : summary.keySet()) {
            String[] parts = key.split("_");
            if (parts.length >= 2) {
                String featureName = parts[0];
                String statType = parts[1];
                
                featureGroups.computeIfAbsent(featureName, k -> new java.util.ArrayList<>()).add(statType);
            }
        }
        
        // Display summary for each feature
        for (Map.Entry<String, java.util.List<String>> entry : featureGroups.entrySet()) {
            String featureName = entry.getKey();
            java.util.List<String> stats = entry.getValue();
            
            System.out.println("\nFeature: " + featureName);
            System.out.println("-".repeat(30));
            
            for (String stat : stats) {
                String key = featureName + "_" + stat;
                Double value = summary.get(key);
                if (value != null) {
                    System.out.printf("%-15s: %.4f%n", stat, value);
                }
            }
        }
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Save explanation as text file
     * 
     * @param explanation feature contributions
     * @param instance the instance being explained
     * @param filename output filename
     */
    public void saveExplanationAsText(Map<String, Double> explanation, double[] instance, String filename) {
        try {
            java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(filename));
            
            writer.println("MODEL EXPLANATION");
            writer.println("=".repeat(50));
            
            // Write instance values
            writer.println("Instance Values:");
            int featureIndex = 0;
            for (Map.Entry<String, Double> entry : explanation.entrySet()) {
                String featureName = entry.getKey();
                double instanceValue = featureIndex < instance.length ? instance[featureIndex] : 0.0;
                writer.printf("%s: %.4f%n", featureName, instanceValue);
                featureIndex++;
            }
            
            // Write feature contributions
            writer.println("\nFeature Contributions:");
            java.util.List<Map.Entry<String, Double>> sortedFeatures = 
                new java.util.ArrayList<>(explanation.entrySet());
            sortedFeatures.sort((a, b) -> Double.compare(Math.abs(b.getValue()), Math.abs(a.getValue())));
            
            for (Map.Entry<String, Double> entry : sortedFeatures) {
                writer.printf("%s: %.4f%n", entry.getKey(), entry.getValue());
            }
            
            // Write total contribution
            double totalContribution = explanation.values().stream().mapToDouble(Double::doubleValue).sum();
            writer.printf("%nTotal Contribution: %.4f%n", totalContribution);
            
            writer.close();
            System.out.println("Explanation saved to: " + filename);
            
        } catch (java.io.IOException e) {
            System.err.println("Error saving explanation: " + e.getMessage());
        }
    }
}

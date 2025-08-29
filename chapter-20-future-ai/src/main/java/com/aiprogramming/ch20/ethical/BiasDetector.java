package com.aiprogramming.ch20.ethical;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Bias Detection System
 * 
 * Implements comprehensive bias detection algorithms to identify various types
 * of bias in datasets and machine learning models. Supports statistical bias,
 * representation bias, measurement bias, and algorithmic bias detection.
 */
public class BiasDetector {
    
    private static final Logger logger = LoggerFactory.getLogger(BiasDetector.class);
    
    private final Map<String, BiasType> biasTypes;
    private final List<BiasReport> biasReports;
    
    public BiasDetector() {
        this.biasTypes = new HashMap<>();
        this.biasReports = new ArrayList<>();
        initializeBiasTypes();
    }
    
    /**
     * Represents different types of bias
     */
    public enum BiasType {
        STATISTICAL_BIAS("Statistical Bias", "Systematic deviation from true values"),
        REPRESENTATION_BIAS("Representation Bias", "Unequal representation of groups in data"),
        MEASUREMENT_BIAS("Measurement Bias", "Systematic errors in data collection"),
        ALGORITHMIC_BIAS("Algorithmic Bias", "Bias introduced by algorithms"),
        SELECTION_BIAS("Selection Bias", "Bias in data selection process"),
        CONFIRMATION_BIAS("Confirmation Bias", "Tendency to favor confirming evidence"),
        ANCHORING_BIAS("Anchoring Bias", "Over-reliance on first piece of information"),
        AVAILABILITY_BIAS("Availability Bias", "Over-estimating probability of memorable events");
        
        private final String name;
        private final String description;
        
        BiasType(String name, String description) {
            this.name = name;
            this.description = description;
        }
        
        public String getName() { return name; }
        public String getDescription() { return description; }
    }
    
    /**
     * Represents a bias report with detailed analysis
     */
    public static class BiasReport {
        private final String datasetName;
        private final BiasType biasType;
        private final double biasScore;
        private final String description;
        private final Map<String, Object> metrics;
        private final List<String> recommendations;
        private final Date timestamp;
        
        public BiasReport(String datasetName, BiasType biasType, double biasScore, 
                         String description, Map<String, Object> metrics, List<String> recommendations) {
            this.datasetName = datasetName;
            this.biasType = biasType;
            this.biasScore = biasScore;
            this.description = description;
            this.metrics = new HashMap<>(metrics);
            this.recommendations = new ArrayList<>(recommendations);
            this.timestamp = new Date();
        }
        
        // Getters
        public String getDatasetName() { return datasetName; }
        public BiasType getBiasType() { return biasType; }
        public double getBiasScore() { return biasScore; }
        public String getDescription() { return description; }
        public Map<String, Object> getMetrics() { return metrics; }
        public List<String> getRecommendations() { return recommendations; }
        public Date getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format("BiasReport{type=%s, score=%.3f, description='%s'}", 
                               biasType.getName(), biasScore, description);
        }
    }
    
    /**
     * Represents a dataset for bias analysis
     */
    public static class Dataset {
        private final String name;
        private final Map<String, List<Object>> features;
        private final List<Object> labels;
        private final Map<String, String> featureTypes; // "numeric", "categorical", "binary"
        
        public Dataset(String name, Map<String, List<Object>> features, List<Object> labels, 
                      Map<String, String> featureTypes) {
            this.name = name;
            this.features = new HashMap<>(features);
            this.labels = new ArrayList<>(labels);
            this.featureTypes = new HashMap<>(featureTypes);
        }
        
        // Getters
        public String getName() { return name; }
        public Map<String, List<Object>> getFeatures() { return features; }
        public List<Object> getLabels() { return labels; }
        public Map<String, String> getFeatureTypes() { return featureTypes; }
        public int getSize() { return labels.size(); }
    }
    
    /**
     * Analyze dataset for various types of bias
     * 
     * @param dataset The dataset to analyze
     * @return List of bias reports
     */
    public List<BiasReport> analyzeDataset(Dataset dataset) {
        logger.info("Starting bias analysis for dataset: {}", dataset.getName());
        
        List<BiasReport> reports = new ArrayList<>();
        
        // Detect different types of bias
        reports.add(detectStatisticalBias(dataset));
        reports.add(detectRepresentationBias(dataset));
        reports.add(detectMeasurementBias(dataset));
        reports.add(detectSelectionBias(dataset));
        
        // Store reports
        biasReports.addAll(reports);
        
        logger.info("Completed bias analysis. Found {} bias issues", reports.size());
        return reports;
    }
    
    /**
     * Detect statistical bias in the dataset
     */
    private BiasReport detectStatisticalBias(Dataset dataset) {
        Map<String, Object> metrics = new HashMap<>();
        List<String> recommendations = new ArrayList<>();
        double biasScore = 0.0;
        
        // Analyze numeric features for statistical bias
        for (Map.Entry<String, List<Object>> entry : dataset.getFeatures().entrySet()) {
            String featureName = entry.getKey();
            List<Object> values = entry.getValue();
            
            if ("numeric".equals(dataset.getFeatureTypes().get(featureName))) {
                DescriptiveStatistics stats = new DescriptiveStatistics();
                for (Object value : values) {
                    if (value instanceof Number) {
                        stats.addValue(((Number) value).doubleValue());
                    }
                }
                
                // Check for skewness (asymmetry)
                double skewness = stats.getSkewness();
                if (Math.abs(skewness) > 1.0) {
                    biasScore += Math.abs(skewness) * 0.1;
                    metrics.put(featureName + "_skewness", skewness);
                    recommendations.add("Consider log transformation for " + featureName);
                }
                
                // Check for outliers
                double outlierThreshold = 3.0;
                long outliers = values.stream()
                    .filter(v -> v instanceof Number)
                    .mapToDouble(v -> ((Number) v).doubleValue())
                    .filter(v -> Math.abs((v - stats.getMean()) / stats.getStandardDeviation()) > outlierThreshold)
                    .count();
                
                if (outliers > values.size() * 0.05) { // More than 5% outliers
                    biasScore += 0.2;
                    metrics.put(featureName + "_outliers", outliers);
                    recommendations.add("Investigate outliers in " + featureName);
                }
            }
        }
        
        String description = biasScore > 0.5 ? 
            "Significant statistical bias detected" : 
            "Minimal statistical bias detected";
        
        return new BiasReport(dataset.getName(), BiasType.STATISTICAL_BIAS, 
                            biasScore, description, metrics, recommendations);
    }
    
    /**
     * Detect representation bias in the dataset
     */
    private BiasReport detectRepresentationBias(Dataset dataset) {
        Map<String, Object> metrics = new HashMap<>();
        List<String> recommendations = new ArrayList<>();
        double biasScore = 0.0;
        
        // Analyze categorical features for representation bias
        for (Map.Entry<String, List<Object>> entry : dataset.getFeatures().entrySet()) {
            String featureName = entry.getKey();
            List<Object> values = entry.getValue();
            
            if ("categorical".equals(dataset.getFeatureTypes().get(featureName))) {
                Map<Object, Long> categoryCounts = values.stream()
                    .collect(Collectors.groupingBy(v -> v, Collectors.counting()));
                
                if (categoryCounts.size() > 1) {
                    // Calculate representation balance
                    double total = categoryCounts.values().stream().mapToLong(Long::longValue).sum();
                    double maxCount = categoryCounts.values().stream().mapToLong(Long::longValue).max().orElse(0);
                    double minCount = categoryCounts.values().stream().mapToLong(Long::longValue).min().orElse(0);
                    
                    double balanceRatio = minCount / maxCount;
                    if (balanceRatio < 0.2) { // Less than 20% balance
                        biasScore += (1.0 - balanceRatio) * 0.5;
                        metrics.put(featureName + "_balance_ratio", balanceRatio);
                        metrics.put(featureName + "_category_counts", categoryCounts);
                        recommendations.add("Consider oversampling underrepresented categories in " + featureName);
                    }
                }
            }
        }
        
        // Check label distribution
        Map<Object, Long> labelCounts = dataset.getLabels().stream()
            .collect(Collectors.groupingBy(l -> l, Collectors.counting()));
        
        if (labelCounts.size() > 1) {
            double total = labelCounts.values().stream().mapToLong(Long::longValue).sum();
            double maxCount = labelCounts.values().stream().mapToLong(Long::longValue).max().orElse(0);
            double minCount = labelCounts.values().stream().mapToLong(Long::longValue).min().orElse(0);
            
            double labelBalanceRatio = minCount / maxCount;
            if (labelBalanceRatio < 0.3) { // Less than 30% balance for labels
                biasScore += (1.0 - labelBalanceRatio) * 0.3;
                metrics.put("label_balance_ratio", labelBalanceRatio);
                metrics.put("label_counts", labelCounts);
                recommendations.add("Consider class balancing techniques for imbalanced labels");
            }
        }
        
        String description = biasScore > 0.5 ? 
            "Significant representation bias detected" : 
            "Minimal representation bias detected";
        
        return new BiasReport(dataset.getName(), BiasType.REPRESENTATION_BIAS, 
                            biasScore, description, metrics, recommendations);
    }
    
    /**
     * Detect measurement bias in the dataset
     */
    private BiasReport detectMeasurementBias(Dataset dataset) {
        Map<String, Object> metrics = new HashMap<>();
        List<String> recommendations = new ArrayList<>();
        double biasScore = 0.0;
        
        // Check for missing values
        for (Map.Entry<String, List<Object>> entry : dataset.getFeatures().entrySet()) {
            String featureName = entry.getKey();
            List<Object> values = entry.getValue();
            
            long missingCount = values.stream().filter(v -> v == null).count();
            double missingRatio = (double) missingCount / values.size();
            
            if (missingRatio > 0.1) { // More than 10% missing
                biasScore += missingRatio * 0.3;
                metrics.put(featureName + "_missing_ratio", missingRatio);
                recommendations.add("Investigate missing values in " + featureName);
            }
        }
        
        // Check for inconsistent data types
        for (Map.Entry<String, List<Object>> entry : dataset.getFeatures().entrySet()) {
            String featureName = entry.getKey();
            List<Object> values = entry.getValue();
            String expectedType = dataset.getFeatureTypes().get(featureName);
            
            if ("numeric".equals(expectedType)) {
                long nonNumericCount = values.stream()
                    .filter(v -> v != null && !(v instanceof Number))
                    .count();
                
                if (nonNumericCount > 0) {
                    biasScore += 0.2;
                    metrics.put(featureName + "_non_numeric_count", nonNumericCount);
                    recommendations.add("Clean non-numeric values in " + featureName);
                }
            }
        }
        
        String description = biasScore > 0.3 ? 
            "Significant measurement bias detected" : 
            "Minimal measurement bias detected";
        
        return new BiasReport(dataset.getName(), BiasType.MEASUREMENT_BIAS, 
                            biasScore, description, metrics, recommendations);
    }
    
    /**
     * Detect selection bias in the dataset
     */
    private BiasReport detectSelectionBias(Dataset dataset) {
        Map<String, Object> metrics = new HashMap<>();
        List<String> recommendations = new ArrayList<>();
        double biasScore = 0.0;
        
        // Check for systematic patterns in data collection
        // This is a simplified implementation - in practice, you'd need domain knowledge
        
        // Check for temporal bias (if timestamps are available)
        // Check for geographic bias (if location data is available)
        // Check for demographic bias (if demographic features are available)
        
        // For now, we'll check for simple patterns
        int size = dataset.getSize();
        if (size < 100) {
            biasScore += 0.3;
            metrics.put("sample_size", size);
            recommendations.add("Consider increasing sample size for better representation");
        }
        
        // Check for random sampling (simplified)
        // In practice, you'd need to know the sampling method
        metrics.put("sampling_method", "unknown");
        recommendations.add("Document and verify sampling methodology");
        
        String description = biasScore > 0.3 ? 
            "Potential selection bias detected" : 
            "No obvious selection bias detected";
        
        return new BiasReport(dataset.getName(), BiasType.SELECTION_BIAS, 
                            biasScore, description, metrics, recommendations);
    }
    
    /**
     * Detect algorithmic bias in model predictions
     * 
     * @param predictions Model predictions
     * @param trueLabels True labels
     * @param sensitiveFeatures Sensitive features for fairness analysis
     * @return Bias report
     */
    public BiasReport detectAlgorithmicBias(List<Double> predictions, List<Object> trueLabels, 
                                           Map<String, List<Object>> sensitiveFeatures) {
        Map<String, Object> metrics = new HashMap<>();
        List<String> recommendations = new ArrayList<>();
        double biasScore = 0.0;
        
        // Calculate overall accuracy
        double accuracy = calculateAccuracy(predictions, trueLabels);
        metrics.put("overall_accuracy", accuracy);
        
        // Analyze bias across sensitive groups
        for (Map.Entry<String, List<Object>> entry : sensitiveFeatures.entrySet()) {
            String featureName = entry.getKey();
            List<Object> values = entry.getValue();
            
            // Group by sensitive feature values
            Map<Object, List<Integer>> groups = new HashMap<>();
            for (int i = 0; i < values.size(); i++) {
                Object value = values.get(i);
                groups.computeIfAbsent(value, k -> new ArrayList<>()).add(i);
            }
            
            // Calculate accuracy for each group
            Map<Object, Double> groupAccuracies = new HashMap<>();
            for (Map.Entry<Object, List<Integer>> group : groups.entrySet()) {
                double groupAccuracy = group.getValue().stream()
                    .mapToDouble(i -> predictions.get(i).equals(trueLabels.get(i)) ? 1.0 : 0.0)
                    .average()
                    .orElse(0.0);
                groupAccuracies.put(group.getKey(), groupAccuracy);
            }
            
            // Check for accuracy disparity
            double maxAccuracy = groupAccuracies.values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            double minAccuracy = groupAccuracies.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            double accuracyDisparity = maxAccuracy - minAccuracy;
            
            if (accuracyDisparity > 0.1) { // More than 10% disparity
                biasScore += accuracyDisparity * 0.5;
                metrics.put(featureName + "_accuracy_disparity", accuracyDisparity);
                metrics.put(featureName + "_group_accuracies", groupAccuracies);
                recommendations.add("Address accuracy disparity across " + featureName + " groups");
            }
        }
        
        String description = biasScore > 0.3 ? 
            "Significant algorithmic bias detected" : 
            "Minimal algorithmic bias detected";
        
        return new BiasReport("Model Predictions", BiasType.ALGORITHMIC_BIAS, 
                            biasScore, description, metrics, recommendations);
    }
    
    /**
     * Calculate accuracy between predictions and true labels
     */
    private double calculateAccuracy(List<Double> predictions, List<Object> trueLabels) {
        if (predictions.size() != trueLabels.size()) {
            throw new IllegalArgumentException("Predictions and labels must have same size");
        }
        
        int correct = 0;
        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.get(i).equals(trueLabels.get(i))) {
                correct++;
            }
        }
        
        return (double) correct / predictions.size();
    }
    
    /**
     * Initialize bias types
     */
    private void initializeBiasTypes() {
        for (BiasType type : BiasType.values()) {
            biasTypes.put(type.name(), type);
        }
    }
    
    /**
     * Get all bias reports
     */
    public List<BiasReport> getBiasReports() {
        return new ArrayList<>(biasReports);
    }
    
    /**
     * Get bias reports by type
     */
    public List<BiasReport> getBiasReportsByType(BiasType type) {
        return biasReports.stream()
            .filter(report -> report.getBiasType() == type)
            .collect(Collectors.toList());
    }
    
    /**
     * Get overall bias score
     */
    public double getOverallBiasScore() {
        return biasReports.stream()
            .mapToDouble(BiasReport::getBiasScore)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Generate comprehensive bias summary
     */
    public String generateBiasSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("=== Bias Analysis Summary ===\n");
        summary.append("Overall Bias Score: ").append(String.format("%.3f", getOverallBiasScore())).append("\n");
        summary.append("Total Bias Issues: ").append(biasReports.size()).append("\n\n");
        
        for (BiasType type : BiasType.values()) {
            List<BiasReport> typeReports = getBiasReportsByType(type);
            if (!typeReports.isEmpty()) {
                summary.append(type.getName()).append(":\n");
                for (BiasReport report : typeReports) {
                    summary.append("  - ").append(report.getDescription())
                           .append(" (Score: ").append(String.format("%.3f", report.getBiasScore())).append(")\n");
                }
                summary.append("\n");
            }
        }
        
        return summary.toString();
    }
}

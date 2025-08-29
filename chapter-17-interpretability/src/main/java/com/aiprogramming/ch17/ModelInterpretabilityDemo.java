package com.aiprogramming.ch17;

import com.aiprogramming.ch17.lime.LIMEExplainer;
import com.aiprogramming.ch17.shap.SHAPExplainer;
import com.aiprogramming.ch17.fairness.FairnessMetrics;
import com.aiprogramming.ch17.fairness.BiasDetector;
import com.aiprogramming.ch17.visualization.FeatureImportancePlot;
import com.aiprogramming.ch17.visualization.FairnessDashboard;
import com.aiprogramming.ch17.visualization.ExplanationVisualizer;
import com.aiprogramming.ch17.utils.ModelWrapper;
import com.aiprogramming.ch17.utils.DataProcessor;

import java.util.*;

/**
 * Main demonstration class for Chapter 17: Model Interpretability and Explainability
 * 
 * This class showcases various techniques for making machine learning models
 * more transparent and understandable, including LIME, SHAP, fairness assessment,
 * and bias detection.
 */
public class ModelInterpretabilityDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 17: Model Interpretability and Explainability ===\n");
        
        try {
            // Initialize data processor and model wrapper
            DataProcessor dataProcessor = new DataProcessor();
            ModelWrapper modelWrapper = new ModelWrapper();
            
            // Load sample data
            System.out.println("1. Loading sample data...");
            double[][] trainingData = dataProcessor.loadSampleData();
            String[] featureNames = dataProcessor.getFeatureNames();
            
            // Train a sample model
            System.out.println("2. Training sample model...");
            modelWrapper.trainModel(trainingData);
            
            // Demonstrate LIME explanations
            demonstrateLIME(modelWrapper, trainingData, featureNames);
            
            // Demonstrate SHAP explanations
            demonstrateSHAP(modelWrapper, trainingData, featureNames);
            
            // Demonstrate fairness assessment
            demonstrateFairness(modelWrapper, trainingData);
            
            // Demonstrate bias detection
            demonstrateBiasDetection(modelWrapper, trainingData);
            
            // Demonstrate visualizations
            demonstrateVisualizations(modelWrapper, trainingData, featureNames);
            
            System.out.println("\n=== All demonstrations completed successfully! ===");
            
        } catch (Exception e) {
            System.err.println("Error during demonstration: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrate LIME (Local Interpretable Model-agnostic Explanations)
     */
    private static void demonstrateLIME(ModelWrapper model, double[][] data, String[] featureNames) {
        System.out.println("\n3. Demonstrating LIME Explanations...");
        
        try {
            LIMEExplainer limeExplainer = new LIMEExplainer(model, data, featureNames);
            
            // Select a sample instance for explanation
            double[] sampleInstance = data[0];
            
            // Generate LIME explanation
            Map<String, Double> explanation = limeExplainer.explain(sampleInstance);
            
            System.out.println("LIME Explanation for sample instance:");
            System.out.println("Feature contributions:");
            explanation.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(entry -> {
                    System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
                });
            
            // Generate explanations for multiple instances
            System.out.println("\nGenerating explanations for multiple instances...");
            List<Map<String, Double>> explanations = new ArrayList<>();
            for (int i = 0; i < Math.min(5, data.length); i++) {
                explanations.add(limeExplainer.explain(data[i]));
            }
            
            System.out.println("Generated " + explanations.size() + " LIME explanations");
            
        } catch (Exception e) {
            System.err.println("Error in LIME demonstration: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate SHAP (SHapley Additive exPlanations)
     */
    private static void demonstrateSHAP(ModelWrapper model, double[][] data, String[] featureNames) {
        System.out.println("\n4. Demonstrating SHAP Explanations...");
        
        try {
            SHAPExplainer shapExplainer = new SHAPExplainer(model, data, featureNames);
            
            // Calculate SHAP values for a sample instance
            double[] sampleInstance = data[0];
            double[] shapValues = shapExplainer.explain(sampleInstance);
            
            System.out.println("SHAP values for sample instance:");
            for (int i = 0; i < featureNames.length; i++) {
                System.out.printf("  %s: %.4f%n", featureNames[i], shapValues[i]);
            }
            
            // Calculate global feature importance
            System.out.println("\nCalculating global feature importance...");
            double[] globalImportance = shapExplainer.calculateGlobalImportance();
            
            System.out.println("Global feature importance:");
            for (int i = 0; i < featureNames.length; i++) {
                System.out.printf("  %s: %.4f%n", featureNames[i], globalImportance[i]);
            }
            
            // Calculate SHAP values for multiple instances
            System.out.println("\nCalculating SHAP values for multiple instances...");
            List<double[]> allShapValues = new ArrayList<>();
            for (int i = 0; i < Math.min(10, data.length); i++) {
                allShapValues.add(shapExplainer.explain(data[i]));
            }
            
            System.out.println("Calculated SHAP values for " + allShapValues.size() + " instances");
            
        } catch (Exception e) {
            System.err.println("Error in SHAP demonstration: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate fairness assessment
     */
    private static void demonstrateFairness(ModelWrapper model, double[][] data) {
        System.out.println("\n5. Demonstrating Fairness Assessment...");
        
        try {
            FairnessMetrics fairnessMetrics = new FairnessMetrics(model, data);
            
            // Simulate sensitive attributes (in real scenario, these would come from data)
            String[] sensitiveAttributes = {"gender", "age_group", "income_level"};
            
            for (String attribute : sensitiveAttributes) {
                System.out.println("\nFairness metrics for " + attribute + ":");
                
                // Calculate demographic parity
                double demographicParity = fairnessMetrics.calculateDemographicParity(attribute);
                System.out.printf("  Demographic Parity: %.4f%n", demographicParity);
                
                // Calculate equalized odds
                double equalizedOdds = fairnessMetrics.calculateEqualizedOdds(attribute);
                System.out.printf("  Equalized Odds: %.4f%n", equalizedOdds);
                
                // Calculate equal opportunity
                double equalOpportunity = fairnessMetrics.calculateEqualOpportunity(attribute);
                System.out.printf("  Equal Opportunity: %.4f%n", equalOpportunity);
                
                // Calculate individual fairness
                double individualFairness = fairnessMetrics.calculateIndividualFairness(attribute);
                System.out.printf("  Individual Fairness: %.4f%n", individualFairness);
            }
            
            // Generate fairness report
            System.out.println("\nGenerating comprehensive fairness report...");
            Map<String, Map<String, Double>> fairnessReport = fairnessMetrics.generateFairnessReport();
            
            System.out.println("Fairness report generated successfully");
            
        } catch (Exception e) {
            System.err.println("Error in fairness demonstration: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate bias detection
     */
    private static void demonstrateBiasDetection(ModelWrapper model, double[][] data) {
        System.out.println("\n6. Demonstrating Bias Detection...");
        
        try {
            BiasDetector biasDetector = new BiasDetector(model, data);
            
            // Detect bias for different sensitive attributes
            String[] sensitiveAttributes = {"gender", "age_group", "income_level"};
            
            for (String attribute : sensitiveAttributes) {
                System.out.println("\nBias detection for " + attribute + ":");
                
                // Detect statistical bias
                double statisticalBias = biasDetector.detectStatisticalBias(attribute);
                System.out.printf("  Statistical Bias: %.4f%n", statisticalBias);
                
                // Detect disparate impact
                double disparateImpact = biasDetector.detectDisparateImpact(attribute);
                System.out.printf("  Disparate Impact: %.4f%n", disparateImpact);
                
                // Detect calibration bias
                double calibrationBias = biasDetector.detectCalibrationBias(attribute);
                System.out.printf("  Calibration Bias: %.4f%n", calibrationBias);
                
                // Generate bias report
                Map<String, Object> biasReport = biasDetector.generateBiasReport(attribute);
                System.out.println("  Bias report generated");
            }
            
            // Detect intersectional bias
            System.out.println("\nDetecting intersectional bias...");
            double intersectionalBias = biasDetector.detectIntersectionalBias(
                new String[]{"gender", "age_group"});
            System.out.printf("Intersectional Bias: %.4f%n", intersectionalBias);
            
        } catch (Exception e) {
            System.err.println("Error in bias detection demonstration: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate visualization features
     */
    private static void demonstrateVisualizations(ModelWrapper model, double[][] data, String[] featureNames) {
        System.out.println("\n7. Demonstrating Visualizations...");
        
        try {
            // Feature importance visualization
            System.out.println("Creating feature importance plot...");
            FeatureImportancePlot importancePlot = new FeatureImportancePlot();
            
            // Generate sample importance scores
            double[] importanceScores = new double[featureNames.length];
            Random random = new Random(42); // Fixed seed for reproducibility
            for (int i = 0; i < importanceScores.length; i++) {
                importanceScores[i] = random.nextDouble();
            }
            
            importancePlot.plotFeatureImportance(featureNames, importanceScores);
            System.out.println("Feature importance plot created");
            
            // Fairness dashboard
            System.out.println("Creating fairness dashboard...");
            FairnessDashboard fairnessDashboard = new FairnessDashboard();
            
            // Add sample fairness metrics
            Map<String, Double> demographicParityScores = new HashMap<>();
            demographicParityScores.put("Group A", 0.85);
            demographicParityScores.put("Group B", 0.82);
            demographicParityScores.put("Group C", 0.88);
            
            fairnessDashboard.addMetric("Demographic Parity", demographicParityScores);
            System.out.println("Fairness dashboard created");
            
            // Explanation visualization
            System.out.println("Creating explanation visualization...");
            ExplanationVisualizer explanationVisualizer = new ExplanationVisualizer();
            
            // Create sample explanation
            Map<String, Double> sampleExplanation = new HashMap<>();
            for (int i = 0; i < Math.min(5, featureNames.length); i++) {
                sampleExplanation.put(featureNames[i], Math.random());
            }
            
            explanationVisualizer.visualizeExplanation(sampleExplanation, data[0]);
            System.out.println("Explanation visualization created");
            
            System.out.println("All visualizations created successfully");
            
        } catch (Exception e) {
            System.err.println("Error in visualization demonstration: " + e.getMessage());
        }
    }
}

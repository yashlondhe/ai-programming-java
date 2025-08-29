package com.aiprogramming.ch17.lime;

import com.aiprogramming.ch17.utils.ModelWrapper;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.*;

/**
 * LIME (Local Interpretable Model-agnostic Explanations) implementation
 * 
 * LIME provides local explanations for individual predictions by approximating
 * the model's behavior around a specific instance using a linear model.
 */
public class LIMEExplainer {
    
    private final ModelWrapper model;
    private final double[][] trainingData;
    private final String[] featureNames;
    private final Random random;
    private final int numSamples;
    private final double kernelWidth;
    
    /**
     * Constructor for LIME explainer
     * 
     * @param model the model to explain
     * @param trainingData training data for generating perturbations
     * @param featureNames names of the features
     */
    public LIMEExplainer(ModelWrapper model, double[][] trainingData, String[] featureNames) {
        this.model = model;
        this.trainingData = trainingData;
        this.featureNames = featureNames;
        this.random = new Random(42); // Fixed seed for reproducibility
        this.numSamples = 1000; // Number of perturbed samples to generate
        this.kernelWidth = 0.25; // Kernel width for distance weighting
    }
    
    /**
     * Generate explanation for a single instance
     * 
     * @param instance the instance to explain
     * @return map of feature names to their contribution scores
     */
    public Map<String, Double> explain(double[] instance) {
        try {
            // Generate perturbed samples around the instance
            double[][] perturbedSamples = generatePerturbedSamples(instance);
            
            // Get predictions for perturbed samples
            double[] predictions = new double[perturbedSamples.length];
            for (int i = 0; i < perturbedSamples.length; i++) {
                predictions[i] = model.predict(perturbedSamples[i]);
            }
            
            // Calculate distances and weights
            double[] distances = calculateDistances(instance, perturbedSamples);
            double[] weights = calculateWeights(distances);
            
            // Fit local linear model
            LocalLinearModel localModel = fitLocalModel(perturbedSamples, predictions, weights);
            
            // Extract feature contributions
            Map<String, Double> featureContributions = new HashMap<>();
            double[] coefficients = localModel.getCoefficients();
            
            for (int i = 0; i < featureNames.length; i++) {
                featureContributions.put(featureNames[i], coefficients[i]);
            }
            
            return featureContributions;
            
        } catch (Exception e) {
            System.err.println("Error generating LIME explanation: " + e.getMessage());
            return new HashMap<>();
        }
    }
    
    /**
     * Generate perturbed samples around the given instance
     */
    private double[][] generatePerturbedSamples(double[] instance) {
        double[][] samples = new double[numSamples][instance.length];
        
        // First sample is the original instance
        samples[0] = instance.clone();
        
        // Generate perturbed samples
        for (int i = 1; i < numSamples; i++) {
            samples[i] = perturbInstance(instance);
        }
        
        return samples;
    }
    
    /**
     * Perturb a single instance by randomly modifying features
     */
    private double[] perturbInstance(double[] instance) {
        double[] perturbed = instance.clone();
        
        // Randomly perturb features
        for (int j = 0; j < instance.length; j++) {
            if (random.nextDouble() < 0.5) { // 50% chance to perturb each feature
                // Add random noise
                double noise = random.nextGaussian() * 0.1;
                perturbed[j] += noise;
                
                // Ensure values stay within reasonable bounds
                perturbed[j] = Math.max(0, Math.min(1, perturbed[j]));
            }
        }
        
        return perturbed;
    }
    
    /**
     * Calculate distances between original instance and perturbed samples
     */
    private double[] calculateDistances(double[] instance, double[][] samples) {
        double[] distances = new double[samples.length];
        
        for (int i = 0; i < samples.length; i++) {
            double distance = 0.0;
            for (int j = 0; j < instance.length; j++) {
                double diff = instance[j] - samples[i][j];
                distance += diff * diff;
            }
            distances[i] = Math.sqrt(distance);
        }
        
        return distances;
    }
    
    /**
     * Calculate weights based on distances using exponential kernel
     */
    private double[] calculateWeights(double[] distances) {
        double[] weights = new double[distances.length];
        
        for (int i = 0; i < distances.length; i++) {
            weights[i] = Math.exp(-distances[i] * distances[i] / (2 * kernelWidth * kernelWidth));
        }
        
        return weights;
    }
    
    /**
     * Fit a local linear model using weighted least squares
     */
    private LocalLinearModel fitLocalModel(double[][] samples, double[] predictions, double[] weights) {
        try {
            // Prepare data for regression
            RealMatrix X = new Array2DRowRealMatrix(samples.length, samples[0].length + 1);
            RealVector y = new ArrayRealVector(predictions);
            RealVector w = new ArrayRealVector(weights);
            
            // Add intercept term and scale by weights
            for (int i = 0; i < samples.length; i++) {
                X.setEntry(i, 0, 1.0); // Intercept
                for (int j = 0; j < samples[i].length; j++) {
                    X.setEntry(i, j + 1, samples[i][j]);
                }
            }
            
            // Weight the data
            RealMatrix XWeighted = X.scalarMultiply(1.0);
            RealVector yWeighted = y.mapMultiply(1.0);
            
            for (int i = 0; i < weights.length; i++) {
                double weight = Math.sqrt(weights[i]);
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    XWeighted.setEntry(i, j, X.getEntry(i, j) * weight);
                }
                yWeighted.setEntry(i, y.getEntry(i) * weight);
            }
            
            // Fit linear regression
            OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
            regression.newSampleData(yWeighted.toArray(), XWeighted.getData());
            
            double[] coefficients = regression.estimateRegressionParameters();
            
            return new LocalLinearModel(coefficients, regression.calculateRSquared());
            
        } catch (Exception e) {
            System.err.println("Error fitting local model: " + e.getMessage());
            // Return a simple model with zeros if regression fails
            double[] coefficients = new double[samples[0].length + 1];
            return new LocalLinearModel(coefficients, 0.0);
        }
    }
    
    /**
     * Generate explanations for multiple instances
     * 
     * @param instances array of instances to explain
     * @return list of explanations
     */
    public List<Map<String, Double>> explainMultiple(double[][] instances) {
        List<Map<String, Double>> explanations = new ArrayList<>();
        
        for (double[] instance : instances) {
            explanations.add(explain(instance));
        }
        
        return explanations;
    }
    
    /**
     * Get the most important features for an explanation
     * 
     * @param explanation the explanation map
     * @param topK number of top features to return
     * @return list of top feature names
     */
    public List<String> getTopFeatures(Map<String, Double> explanation, int topK) {
        return explanation.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(Map.Entry::getKey)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Calculate explanation stability by generating multiple explanations
     * 
     * @param instance the instance to explain
     * @param numRuns number of explanation runs
     * @return standard deviation of feature contributions
     */
    public Map<String, Double> calculateStability(double[] instance, int numRuns) {
        Map<String, Double> meanContributions = new HashMap<>();
        Map<String, Double> sumSquares = new HashMap<>();
        Map<String, Integer> counts = new HashMap<>();
        
        // Initialize
        for (String feature : featureNames) {
            meanContributions.put(feature, 0.0);
            sumSquares.put(feature, 0.0);
            counts.put(feature, 0);
        }
        
        // Generate multiple explanations
        for (int run = 0; run < numRuns; run++) {
            Map<String, Double> explanation = explain(instance);
            
            for (Map.Entry<String, Double> entry : explanation.entrySet()) {
                String feature = entry.getKey();
                double value = entry.getValue();
                
                double currentMean = meanContributions.get(feature);
                int currentCount = counts.get(feature);
                
                // Update mean incrementally
                double newMean = (currentMean * currentCount + value) / (currentCount + 1);
                meanContributions.put(feature, newMean);
                
                // Update sum of squares
                sumSquares.put(feature, sumSquares.get(feature) + value * value);
                counts.put(feature, currentCount + 1);
            }
        }
        
        // Calculate standard deviations
        Map<String, Double> standardDeviations = new HashMap<>();
        for (String feature : featureNames) {
            int count = counts.get(feature);
            if (count > 1) {
                double mean = meanContributions.get(feature);
                double variance = (sumSquares.get(feature) - count * mean * mean) / (count - 1);
                standardDeviations.put(feature, Math.sqrt(variance));
            } else {
                standardDeviations.put(feature, 0.0);
            }
        }
        
        return standardDeviations;
    }
}

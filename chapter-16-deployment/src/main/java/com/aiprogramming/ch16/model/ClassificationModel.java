package com.aiprogramming.ch16.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.persistence.DiscriminatorValue;
import javax.persistence.Entity;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Concrete implementation of AIModel for classification tasks
 * 
 * This class demonstrates how to implement a specific model type
 * with its own prediction logic and model management
 */
@Entity
@DiscriminatorValue("CLASSIFICATION")
public class ClassificationModel extends AIModel {

    private static final Logger logger = LoggerFactory.getLogger(ClassificationModel.class);
    
    // In-memory model storage (in production, this would be external)
    private static final Map<String, ClassificationModel> loadedModels = new ConcurrentHashMap<>();
    
    // Model parameters (simplified for demonstration)
    private double[] weights;
    private double bias;
    private List<String> featureNames;
    private List<String> classLabels;
    
    // Model state
    private boolean modelLoaded = false;
    private long lastPredictionTime = 0;
    private long totalPredictions = 0;
    private double averagePredictionTime = 0.0;

    public ClassificationModel() {
        super();
    }

    public ClassificationModel(String modelId, String name, String version, String description) {
        super(modelId, name, version, description);
    }

    @Override
    @JsonIgnore
    public String getModelType() {
        return "CLASSIFICATION";
    }

    @Override
    @JsonIgnore
    public boolean isLoaded() {
        return modelLoaded && loadedModels.containsKey(getModelId());
    }

    @Override
    @JsonIgnore
    public void loadModel() {
        if (isLoaded()) {
            logger.info("Model {} is already loaded", getModelId());
            return;
        }

        try {
            logger.info("Loading classification model: {}", getModelId());
            
            // Simulate model loading (in real implementation, load from file/database)
            initializeModel();
            
            // Register the loaded model
            loadedModels.put(getModelId(), this);
            modelLoaded = true;
            
            logger.info("Successfully loaded model: {}", getModelId());
            
        } catch (Exception e) {
            logger.error("Failed to load model: {}", getModelId(), e);
            throw new RuntimeException("Model loading failed", e);
        }
    }

    @Override
    @JsonIgnore
    public void unloadModel() {
        if (!isLoaded()) {
            logger.info("Model {} is not loaded", getModelId());
            return;
        }

        try {
            logger.info("Unloading classification model: {}", getModelId());
            
            // Clean up model resources
            weights = null;
            bias = 0.0;
            featureNames = null;
            classLabels = null;
            
            // Remove from loaded models
            loadedModels.remove(getModelId());
            modelLoaded = false;
            
            logger.info("Successfully unloaded model: {}", getModelId());
            
        } catch (Exception e) {
            logger.error("Failed to unload model: {}", getModelId(), e);
            throw new RuntimeException("Model unloading failed", e);
        }
    }

    /**
     * Make a prediction using the loaded model
     * 
     * @param features input features for prediction
     * @return prediction result
     */
    public ClassificationPrediction predict(double[] features) {
        if (!isLoaded()) {
            throw new IllegalStateException("Model is not loaded: " + getModelId());
        }

        long startTime = System.currentTimeMillis();
        
        try {
            // Validate input
            if (features == null || features.length != weights.length) {
                throw new IllegalArgumentException(
                    String.format("Expected %d features, got %d", 
                        weights.length, features == null ? 0 : features.length));
            }

            // Simple linear classification (for demonstration)
            double score = bias;
            for (int i = 0; i < features.length; i++) {
                score += weights[i] * features[i];
            }

            // Convert to probability using sigmoid
            double probability = 1.0 / (1.0 + Math.exp(-score));
            
            // Determine predicted class
            String predictedClass = probability > 0.5 ? classLabels.get(1) : classLabels.get(0);
            
            // Create prediction result
            ClassificationPrediction prediction = new ClassificationPrediction(
                predictedClass,
                probability,
                features,
                System.currentTimeMillis()
            );

            // Update statistics
            updatePredictionStats(System.currentTimeMillis() - startTime);
            
            return prediction;
            
        } catch (Exception e) {
            logger.error("Prediction failed for model: {}", getModelId(), e);
            throw new RuntimeException("Prediction failed", e);
        }
    }

    /**
     * Initialize model with sample data (for demonstration)
     */
    private void initializeModel() {
        // Simulate loading model parameters
        this.weights = new double[]{0.5, -0.3, 0.8, -0.2, 0.1};
        this.bias = -0.1;
        this.featureNames = List.of("feature1", "feature2", "feature3", "feature4", "feature5");
        this.classLabels = List.of("negative", "positive");
        
        // Update metadata
        addMetadata("algorithm", "logistic_regression");
        addMetadata("feature_count", String.valueOf(weights.length));
        addMetadata("class_count", String.valueOf(classLabels.size()));
    }

    /**
     * Update prediction statistics
     */
    private void updatePredictionStats(long predictionTime) {
        this.lastPredictionTime = System.currentTimeMillis();
        this.totalPredictions++;
        
        // Update average prediction time
        this.averagePredictionTime = 
            ((this.averagePredictionTime * (this.totalPredictions - 1)) + predictionTime) / this.totalPredictions;
    }

    /**
     * Get model performance statistics
     */
    public ModelStats getModelStats() {
        return new ModelStats(
            totalPredictions,
            averagePredictionTime,
            lastPredictionTime,
            getAccuracy()
        );
    }

    // Getters and Setters
    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }

    public void setFeatureNames(List<String> featureNames) {
        this.featureNames = featureNames;
    }

    public List<String> getClassLabels() {
        return classLabels;
    }

    public void setClassLabels(List<String> classLabels) {
        this.classLabels = classLabels;
    }

    public long getLastPredictionTime() {
        return lastPredictionTime;
    }

    public long getTotalPredictions() {
        return totalPredictions;
    }

    public double getAveragePredictionTime() {
        return averagePredictionTime;
    }

    /**
     * Get all currently loaded models
     */
    public static Map<String, ClassificationModel> getLoadedModels() {
        return new ConcurrentHashMap<>(loadedModels);
    }
}

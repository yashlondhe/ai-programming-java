package com.aiprogramming.ch16.service;

import com.aiprogramming.ch16.model.AIModel;
import com.aiprogramming.ch16.model.ClassificationModel;
import com.aiprogramming.ch16.model.ClassificationPrediction;
import com.aiprogramming.utils.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;
import java.util.UUID;

/**
 * Service class for handling model predictions and inference
 */
@Service
public class PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(PredictionService.class);

    @Autowired
    private ModelService modelService;

    /**
     * Make a prediction using a deployed model
     * 
     * @param modelId the ID of the model to use for prediction
     * @param features input features for the prediction
     * @return prediction result
     */
    public ClassificationPrediction predict(String modelId, double[] features) {
        logger.info("Making prediction with model: {}", modelId);
        
        // Using ValidationUtils for input validation
        try {
            ValidationUtils.validateNotNull(modelId, "modelId");
            ValidationUtils.validateNonEmptyString(modelId, "modelId");
            ValidationUtils.validateVector(features, "features");
            ValidationUtils.validateFiniteValues(features, "features");
        } catch (IllegalArgumentException e) {
            logger.error("Input validation failed: {}", e.getMessage());
            throw e;
        }
        
        // Get the model
        Optional<AIModel> modelOpt = modelService.getModelById(modelId);
        if (modelOpt.isEmpty()) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }
        
        AIModel model = modelOpt.get();
        
        // Check if model is active (deployed or A/B testing)
        if (!model.getStatus().isActive()) {
            throw new IllegalStateException(
                "Model is not active. Current status: " + model.getStatus());
        }
        
        // Check if model is loaded
        if (!model.isLoaded()) {
            logger.warn("Model {} is not loaded, attempting to load it", modelId);
            try {
                model.loadModel();
            } catch (Exception e) {
                throw new RuntimeException("Failed to load model for prediction", e);
            }
        }
        
        // Make prediction based on model type
        if (model instanceof ClassificationModel) {
            return predictWithClassificationModel((ClassificationModel) model, features);
        } else {
            throw new UnsupportedOperationException(
                "Prediction not supported for model type: " + model.getModelType());
        }
    }

    /**
     * Make a prediction with a classification model
     */
    private ClassificationPrediction predictWithClassificationModel(
            ClassificationModel model, double[] features) {
        
        long startTime = System.currentTimeMillis();
        
        try {
            // Make the prediction
            ClassificationPrediction prediction = model.predict(features);
            
            // Set additional metadata
            prediction.setModelId(model.getModelId());
            prediction.setPredictionId(generatePredictionId());
            prediction.setProcessingTimeMs(System.currentTimeMillis() - startTime);
            
            logger.info("Prediction completed for model {} in {}ms", 
                model.getModelId(), prediction.getProcessingTimeMs());
            
            return prediction;
            
        } catch (Exception e) {
            logger.error("Prediction failed for model: {}", model.getModelId(), e);
            throw new RuntimeException("Prediction failed", e);
        }
    }

    /**
     * Make a batch prediction using multiple models (A/B testing)
     * 
     * @param modelIds list of model IDs to use for prediction
     * @param features input features for the prediction
     * @return list of predictions from each model
     */
    public java.util.List<ClassificationPrediction> batchPredict(
            java.util.List<String> modelIds, double[] features) {
        
        logger.info("Making batch prediction with {} models", modelIds.size());
        
        return modelIds.stream()
            .map(modelId -> {
                try {
                    return predict(modelId, features);
                } catch (Exception e) {
                    logger.error("Failed to get prediction from model: {}", modelId, e);
                    return null;
                }
            })
            .filter(prediction -> prediction != null)
            .collect(java.util.stream.Collectors.toList());
    }

    /**
     * Get prediction statistics for a model
     * 
     * @param modelId the model ID
     * @return prediction statistics
     */
    public com.aiprogramming.ch16.model.ModelStats getPredictionStats(String modelId) {
        return modelService.getModelStats(modelId);
    }

    /**
     * Validate prediction input
     * 
     * @param features input features to validate
     * @param expectedFeatureCount expected number of features
     */
    public void validatePredictionInput(double[] features, int expectedFeatureCount) {
        if (features == null) {
            throw new IllegalArgumentException("Features cannot be null");
        }
        
        if (features.length != expectedFeatureCount) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", 
                    expectedFeatureCount, features.length));
        }
        
        // Check for NaN or infinite values
        for (int i = 0; i < features.length; i++) {
            if (Double.isNaN(features[i]) || Double.isInfinite(features[i])) {
                throw new IllegalArgumentException(
                    String.format("Invalid feature value at index %d: %f", i, features[i]));
            }
        }
    }

    /**
     * Generate unique prediction ID
     */
    private String generatePredictionId() {
        return "pred_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    }
}

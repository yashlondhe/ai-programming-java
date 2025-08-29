package com.aiprogramming.ch16.controller;

import com.aiprogramming.ch16.model.ClassificationPrediction;
import com.aiprogramming.ch16.model.ModelStats;
import com.aiprogramming.ch16.service.PredictionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;

/**
 * REST API controller for model prediction and inference operations
 */
@RestController
@RequestMapping("/api/v1/predictions")
@CrossOrigin(origins = "*")
public class PredictionController {

    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);

    @Autowired
    private PredictionService predictionService;

    /**
     * Prediction request DTO
     */
    public static class PredictionRequest {
        private double[] features;

        public PredictionRequest() {}

        public PredictionRequest(double[] features) {
            this.features = features;
        }

        public double[] getFeatures() {
            return features;
        }

        public void setFeatures(double[] features) {
            this.features = features;
        }
    }

    /**
     * Batch prediction request DTO
     */
    public static class BatchPredictionRequest {
        private List<String> modelIds;
        private double[] features;

        public BatchPredictionRequest() {}

        public BatchPredictionRequest(List<String> modelIds, double[] features) {
            this.modelIds = modelIds;
            this.features = features;
        }

        public List<String> getModelIds() {
            return modelIds;
        }

        public void setModelIds(List<String> modelIds) {
            this.modelIds = modelIds;
        }

        public double[] getFeatures() {
            return features;
        }

        public void setFeatures(double[] features) {
            this.features = features;
        }
    }

    /**
     * Make a prediction using a specific model
     * 
     * POST /api/v1/predictions/{modelId}
     */
    @PostMapping("/{modelId}")
    public ResponseEntity<ClassificationPrediction> predict(
            @PathVariable String modelId,
            @Valid @RequestBody PredictionRequest request) {
        
        logger.info("Making prediction with model: {}", modelId);
        
        try {
            // Validate input
            if (request.getFeatures() == null) {
                return ResponseEntity.badRequest().build();
            }

            // Make prediction
            ClassificationPrediction prediction = predictionService.predict(modelId, request.getFeatures());
            return ResponseEntity.ok(prediction);
            
        } catch (IllegalArgumentException e) {
            logger.error("Invalid prediction request: {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (IllegalStateException e) {
            logger.error("Model not ready for prediction: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).build();
        } catch (Exception e) {
            logger.error("Prediction failed for model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Make a batch prediction using multiple models (A/B testing)
     * 
     * POST /api/v1/predictions/batch
     */
    @PostMapping("/batch")
    public ResponseEntity<List<ClassificationPrediction>> batchPredict(
            @Valid @RequestBody BatchPredictionRequest request) {
        
        logger.info("Making batch prediction with {} models", 
            request.getModelIds() != null ? request.getModelIds().size() : 0);
        
        try {
            // Validate input
            if (request.getModelIds() == null || request.getModelIds().isEmpty()) {
                return ResponseEntity.badRequest().build();
            }
            if (request.getFeatures() == null) {
                return ResponseEntity.badRequest().build();
            }

            // Make batch prediction
            List<ClassificationPrediction> predictions = predictionService.batchPredict(
                request.getModelIds(), request.getFeatures());
            return ResponseEntity.ok(predictions);
            
        } catch (Exception e) {
            logger.error("Batch prediction failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get prediction statistics for a model
     * 
     * GET /api/v1/predictions/{modelId}/stats
     */
    @GetMapping("/{modelId}/stats")
    public ResponseEntity<ModelStats> getPredictionStats(@PathVariable String modelId) {
        logger.info("Getting prediction stats for model: {}", modelId);
        
        try {
            ModelStats stats = predictionService.getPredictionStats(modelId);
            return ResponseEntity.ok(stats);
        } catch (IllegalArgumentException e) {
            logger.error("Model not found: {}", modelId);
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            logger.error("Failed to get prediction stats for model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Health check endpoint for prediction service
     * 
     * GET /api/v1/predictions/health
     */
    @GetMapping("/health")
    public ResponseEntity<String> healthCheck() {
        logger.info("Health check request");
        return ResponseEntity.ok("Prediction service is healthy");
    }

    /**
     * Get supported model types
     * 
     * GET /api/v1/predictions/supported-types
     */
    @GetMapping("/supported-types")
    public ResponseEntity<List<String>> getSupportedModelTypes() {
        logger.info("Getting supported model types");
        
        List<String> supportedTypes = List.of("CLASSIFICATION");
        return ResponseEntity.ok(supportedTypes);
    }
}

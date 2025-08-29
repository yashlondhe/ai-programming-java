package com.aiprogramming.ch16.controller;

import com.aiprogramming.ch16.model.AIModel;
import com.aiprogramming.ch16.model.ModelStatus;
import com.aiprogramming.ch16.model.ModelStats;
import com.aiprogramming.ch16.service.ModelService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;
import java.util.Optional;

/**
 * REST API controller for AI model management operations
 */
@RestController
@RequestMapping("/api/v1/models")
@CrossOrigin(origins = "*")
public class ModelController {

    private static final Logger logger = LoggerFactory.getLogger(ModelController.class);

    @Autowired
    private ModelService modelService;

    /**
     * Create a new AI model
     * 
     * POST /api/v1/models
     */
    @PostMapping
    public ResponseEntity<AIModel> createModel(@Valid @RequestBody AIModel model) {
        logger.info("Creating new model: {}", model.getName());
        
        try {
            AIModel createdModel = modelService.createModel(model);
            return ResponseEntity.status(HttpStatus.CREATED).body(createdModel);
        } catch (Exception e) {
            logger.error("Failed to create model", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get all models
     * 
     * GET /api/v1/models
     */
    @GetMapping
    public ResponseEntity<List<AIModel>> getAllModels() {
        logger.info("Getting all models");
        
        try {
            List<AIModel> models = modelService.getAllModels();
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            logger.error("Failed to get models", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get model by ID
     * 
     * GET /api/v1/models/{modelId}
     */
    @GetMapping("/{modelId}")
    public ResponseEntity<AIModel> getModelById(@PathVariable String modelId) {
        logger.info("Getting model by ID: {}", modelId);
        
        try {
            Optional<AIModel> model = modelService.getModelById(modelId);
            return model.map(ResponseEntity::ok)
                       .orElse(ResponseEntity.notFound().build());
        } catch (Exception e) {
            logger.error("Failed to get model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get models by status
     * 
     * GET /api/v1/models/status/{status}
     */
    @GetMapping("/status/{status}")
    public ResponseEntity<List<AIModel>> getModelsByStatus(@PathVariable ModelStatus status) {
        logger.info("Getting models by status: {}", status);
        
        try {
            List<AIModel> models = modelService.getModelsByStatus(status);
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            logger.error("Failed to get models by status: {}", status, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get active models (deployed or A/B testing)
     * 
     * GET /api/v1/models/active
     */
    @GetMapping("/active")
    public ResponseEntity<List<AIModel>> getActiveModels() {
        logger.info("Getting active models");
        
        try {
            List<AIModel> models = modelService.getActiveModels();
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            logger.error("Failed to get active models", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Update model status
     * 
     * PUT /api/v1/models/{modelId}/status
     */
    @PutMapping("/{modelId}/status")
    public ResponseEntity<AIModel> updateModelStatus(
            @PathVariable String modelId, 
            @RequestParam ModelStatus status) {
        logger.info("Updating model {} status to: {}", modelId, status);
        
        try {
            AIModel updatedModel = modelService.updateModelStatus(modelId, status);
            return ResponseEntity.ok(updatedModel);
        } catch (IllegalArgumentException e) {
            logger.error("Invalid request: {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            logger.error("Failed to update model status: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Deploy a model
     * 
     * POST /api/v1/models/{modelId}/deploy
     */
    @PostMapping("/{modelId}/deploy")
    public ResponseEntity<String> deployModel(@PathVariable String modelId) {
        logger.info("Deploying model: {}", modelId);
        
        try {
            boolean success = modelService.deployModel(modelId);
            if (success) {
                return ResponseEntity.ok("Model deployed successfully");
            } else {
                return ResponseEntity.badRequest().body("Failed to deploy model");
            }
        } catch (Exception e) {
            logger.error("Failed to deploy model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Undeploy a model
     * 
     * POST /api/v1/models/{modelId}/undeploy
     */
    @PostMapping("/{modelId}/undeploy")
    public ResponseEntity<String> undeployModel(@PathVariable String modelId) {
        logger.info("Undeploying model: {}", modelId);
        
        try {
            boolean success = modelService.undeployModel(modelId);
            if (success) {
                return ResponseEntity.ok("Model undeployed successfully");
            } else {
                return ResponseEntity.badRequest().body("Failed to undeploy model");
            }
        } catch (Exception e) {
            logger.error("Failed to undeploy model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get model statistics
     * 
     * GET /api/v1/models/{modelId}/stats
     */
    @GetMapping("/{modelId}/stats")
    public ResponseEntity<ModelStats> getModelStats(@PathVariable String modelId) {
        logger.info("Getting stats for model: {}", modelId);
        
        try {
            ModelStats stats = modelService.getModelStats(modelId);
            return ResponseEntity.ok(stats);
        } catch (IllegalArgumentException e) {
            logger.error("Model not found: {}", modelId);
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            logger.error("Failed to get model stats: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Delete a model
     * 
     * DELETE /api/v1/models/{modelId}
     */
    @DeleteMapping("/{modelId}")
    public ResponseEntity<String> deleteModel(@PathVariable String modelId) {
        logger.info("Deleting model: {}", modelId);
        
        try {
            boolean success = modelService.deleteModel(modelId);
            if (success) {
                return ResponseEntity.ok("Model deleted successfully");
            } else {
                return ResponseEntity.badRequest().body("Failed to delete model");
            }
        } catch (Exception e) {
            logger.error("Failed to delete model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Search models by name
     * 
     * GET /api/v1/models/search?name={name}
     */
    @GetMapping("/search")
    public ResponseEntity<List<AIModel>> searchModels(@RequestParam String name) {
        logger.info("Searching models by name: {}", name);
        
        try {
            List<AIModel> models = modelService.searchModelsByName(name);
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            logger.error("Failed to search models", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get models with accuracy above threshold
     * 
     * GET /api/v1/models/accuracy?minAccuracy={minAccuracy}
     */
    @GetMapping("/accuracy")
    public ResponseEntity<List<AIModel>> getModelsByAccuracy(
            @RequestParam double minAccuracy) {
        logger.info("Getting models with accuracy >= {}", minAccuracy);
        
        try {
            List<AIModel> models = modelService.getModelsByAccuracyThreshold(minAccuracy);
            return ResponseEntity.ok(models);
        } catch (Exception e) {
            logger.error("Failed to get models by accuracy", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}

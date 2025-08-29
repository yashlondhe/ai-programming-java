package com.aiprogramming.ch16.service;

import com.aiprogramming.ch16.model.AIModel;
import com.aiprogramming.ch16.model.ClassificationModel;
import com.aiprogramming.ch16.model.ModelStatus;
import com.aiprogramming.ch16.model.ModelStats;
import com.aiprogramming.ch16.repository.ModelRepository;
import com.aiprogramming.utils.ValidationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Service class for AI model management operations
 */
@Service
@Transactional
public class ModelService {

    private static final Logger logger = LoggerFactory.getLogger(ModelService.class);

    @Autowired
    private ModelRepository modelRepository;

    /**
     * Create a new AI model
     */
    public AIModel createModel(AIModel model) {
        logger.info("Creating new model: {}", model.getName());
        
        ValidationUtils.validateNotNull(model, "model");
        ValidationUtils.validateNotNull(model.getName(), "model.name");
        ValidationUtils.validateNonEmptyString(model.getName(), "model.name");
        
        // Generate unique model ID if not provided
        if (model.getModelId() == null || model.getModelId().isEmpty()) {
            model.setModelId(generateModelId());
        }
        
        // Set initial status
        model.setStatus(ModelStatus.DRAFT);
        model.setCreatedAt(LocalDateTime.now());
        model.setUpdatedAt(LocalDateTime.now());
        
        // Save to database
        AIModel savedModel = modelRepository.save(model);
        logger.info("Successfully created model: {}", savedModel.getModelId());
        
        return savedModel;
    }

    /**
     * Get model by ID
     */
    @Transactional(readOnly = true)
    public Optional<AIModel> getModelById(String modelId) {
        ValidationUtils.validateNotNull(modelId, "modelId");
        ValidationUtils.validateNonEmptyString(modelId, "modelId");
        return modelRepository.findByModelId(modelId);
    }

    /**
     * Get all models
     */
    @Transactional(readOnly = true)
    public List<AIModel> getAllModels() {
        return modelRepository.findAll();
    }

    /**
     * Get models by status
     */
    @Transactional(readOnly = true)
    public List<AIModel> getModelsByStatus(ModelStatus status) {
        ValidationUtils.validateNotNull(status, "status");
        return modelRepository.findByStatus(status);
    }

    /**
     * Get active models (deployed or A/B testing)
     */
    @Transactional(readOnly = true)
    public List<AIModel> getActiveModels() {
        return modelRepository.findActiveModels();
    }

    /**
     * Update model status
     */
    public AIModel updateModelStatus(String modelId, ModelStatus newStatus) {
        logger.info("Updating model {} status to: {}", modelId, newStatus);
        
        ValidationUtils.validateNotNull(modelId, "modelId");
        ValidationUtils.validateNonEmptyString(modelId, "modelId");
        ValidationUtils.validateNotNull(newStatus, "newStatus");
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }
        
        AIModel model = modelOpt.get();
        model.setStatus(newStatus);
        model.setUpdatedAt(LocalDateTime.now());
        
        // Handle status-specific logic
        if (newStatus == ModelStatus.DEPLOYED) {
            handleModelDeployment(model);
        } else if (newStatus == ModelStatus.ARCHIVED) {
            handleModelArchival(model);
        }
        
        AIModel updatedModel = modelRepository.save(model);
        logger.info("Successfully updated model {} status to: {}", modelId, newStatus);
        
        return updatedModel;
    }

    /**
     * Deploy a model (load it into memory and make it available for predictions)
     */
    public boolean deployModel(String modelId) {
        logger.info("Deploying model: {}", modelId);
        
        ValidationUtils.validateNotNull(modelId, "modelId");
        ValidationUtils.validateNonEmptyString(modelId, "modelId");
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            logger.error("Model not found for deployment: {}", modelId);
            return false;
        }
        
        AIModel model = modelOpt.get();
        
        // Check if model can be deployed
        if (!model.getStatus().canDeploy()) {
            logger.error("Model {} cannot be deployed in current status: {}", 
                modelId, model.getStatus());
            return false;
        }
        
        try {
            // Load the model into memory
            model.loadModel();
            
            // Update status to deployed
            model.setStatus(ModelStatus.DEPLOYED);
            model.setUpdatedAt(LocalDateTime.now());
            modelRepository.save(model);
            
            logger.info("Successfully deployed model: {}", modelId);
            return true;
            
        } catch (Exception e) {
            logger.error("Failed to deploy model: {}", modelId, e);
            return false;
        }
    }

    /**
     * Undeploy a model (unload from memory)
     */
    public boolean undeployModel(String modelId) {
        logger.info("Undeploying model: {}", modelId);
        
        ValidationUtils.validateNotNull(modelId, "modelId");
        ValidationUtils.validateNonEmptyString(modelId, "modelId");
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            logger.error("Model not found for undeployment: {}", modelId);
            return false;
        }
        
        AIModel model = modelOpt.get();
        
        try {
            // Unload the model from memory
            model.unloadModel();
            
            // Update status to approved (ready for deployment)
            model.setStatus(ModelStatus.APPROVED);
            model.setUpdatedAt(LocalDateTime.now());
            modelRepository.save(model);
            
            logger.info("Successfully undeployed model: {}", modelId);
            return true;
            
        } catch (Exception e) {
            logger.error("Failed to undeploy model: {}", modelId, e);
            return false;
        }
    }

    /**
     * Get model statistics
     */
    @Transactional(readOnly = true)
    public ModelStats getModelStats(String modelId) {
        ValidationUtils.validateNotNull(modelId, "modelId");
        ValidationUtils.validateNonEmptyString(modelId, "modelId");
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }
        
        AIModel model = modelOpt.get();
        
        // For classification models, get detailed stats
        if (model instanceof ClassificationModel) {
            ClassificationModel classificationModel = (ClassificationModel) model;
            return classificationModel.getModelStats();
        }
        
        // For other model types, return basic stats
        return new ModelStats(0, 0.0, System.currentTimeMillis(), null);
    }

    /**
     * Delete a model
     */
    public boolean deleteModel(String modelId) {
        logger.info("Deleting model: {}", modelId);
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            logger.error("Model not found for deletion: {}", modelId);
            return false;
        }
        
        AIModel model = modelOpt.get();
        
        // Check if model can be deleted
        if (model.getStatus().isActive()) {
            logger.error("Cannot delete active model: {}", modelId);
            return false;
        }
        
        try {
            // Unload model if it's loaded
            if (model.isLoaded()) {
                model.unloadModel();
            }
            
            // Delete from database
            modelRepository.delete(model);
            
            logger.info("Successfully deleted model: {}", modelId);
            return true;
            
        } catch (Exception e) {
            logger.error("Failed to delete model: {}", modelId, e);
            return false;
        }
    }

    /**
     * Search models by name
     */
    @Transactional(readOnly = true)
    public List<AIModel> searchModelsByName(String name) {
        return modelRepository.findByNameContainingIgnoreCase(name);
    }

    /**
     * Get models with accuracy above threshold
     */
    @Transactional(readOnly = true)
    public List<AIModel> getModelsByAccuracyThreshold(double minAccuracy) {
        return modelRepository.findByAccuracyAbove(minAccuracy);
    }

    /**
     * Generate unique model ID
     */
    private String generateModelId() {
        return "model_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
    }

    /**
     * Handle model deployment logic
     */
    private void handleModelDeployment(AIModel model) {
        logger.info("Handling deployment for model: {}", model.getModelId());
        
        // In a real implementation, you might:
        // - Load the model into a model serving infrastructure
        // - Update load balancer configuration
        // - Send notifications
        // - Update monitoring dashboards
    }

    /**
     * Handle model archival logic
     */
    private void handleModelArchival(AIModel model) {
        logger.info("Handling archival for model: {}", model.getModelId());
        
        // In a real implementation, you might:
        // - Unload the model from serving infrastructure
        // - Archive model files to long-term storage
        // - Update documentation
        // - Send notifications
    }
}

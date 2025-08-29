package com.aiprogramming.ch16.model;

/**
 * Enum representing the lifecycle status of an AI model
 */
public enum ModelStatus {
    
    /**
     * Model is in development/testing phase
     */
    DRAFT("Draft"),
    
    /**
     * Model is being trained
     */
    TRAINING("Training"),
    
    /**
     * Model training completed, ready for evaluation
     */
    TRAINED("Trained"),
    
    /**
     * Model is being evaluated
     */
    EVALUATING("Evaluating"),
    
    /**
     * Model passed evaluation, ready for deployment
     */
    APPROVED("Approved"),
    
    /**
     * Model is deployed and serving predictions
     */
    DEPLOYED("Deployed"),
    
    /**
     * Model is in A/B testing
     */
    AB_TESTING("A/B Testing"),
    
    /**
     * Model is deprecated and will be removed
     */
    DEPRECATED("Deprecated"),
    
    /**
     * Model is archived and no longer in use
     */
    ARCHIVED("Archived"),
    
    /**
     * Model failed evaluation or deployment
     */
    FAILED("Failed");

    private final String displayName;

    ModelStatus(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Check if the model is in an active state (can serve predictions)
     */
    public boolean isActive() {
        return this == DEPLOYED || this == AB_TESTING;
    }

    /**
     * Check if the model can be deployed
     */
    public boolean canDeploy() {
        return this == APPROVED || this == DEPRECATED;
    }

    /**
     * Check if the model is in a terminal state
     */
    public boolean isTerminal() {
        return this == ARCHIVED || this == FAILED;
    }
}

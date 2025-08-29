package com.aiprogramming.ch18;

import java.util.Map;

/**
 * Represents a single hyperparameter trial
 */
public class HyperparameterTrial {
    
    private final Map<String, Object> parameters;
    private final double score;
    
    public HyperparameterTrial(Map<String, Object> parameters, double score) {
        this.parameters = parameters;
        this.score = score;
    }
    
    public Map<String, Object> getParameters() {
        return parameters;
    }
    
    public double getScore() {
        return score;
    }
    
    @Override
    public String toString() {
        return String.format("Trial{score=%.4f, params=%s}", score, parameters);
    }
}

package com.aiprogramming.ch18;

import java.util.*;

/**
 * Result of hyperparameter optimization
 */
public class HyperparameterOptimizationResult {
    
    private final Map<String, Object> bestParameters;
    private final double bestScore;
    private final List<HyperparameterTrial> trials;
    
    public HyperparameterOptimizationResult(Map<String, Object> bestParameters, 
                                          double bestScore, 
                                          List<HyperparameterTrial> trials) {
        this.bestParameters = bestParameters;
        this.bestScore = bestScore;
        this.trials = trials;
    }
    
    public Map<String, Object> getBestParameters() {
        return bestParameters;
    }
    
    public double getBestScore() {
        return bestScore;
    }
    
    public List<HyperparameterTrial> getTrials() {
        return trials;
    }
    
    /**
     * Print summary of optimization results
     */
    public void printSummary() {
        System.out.println("Best Score: " + String.format("%.4f", bestScore));
        System.out.println("Best Parameters: " + bestParameters);
        System.out.println("Total Trials: " + trials.size());
        
        // Show top 5 trials
        System.out.println("\nTop 5 Trials:");
        trials.stream()
              .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
              .limit(5)
              .forEach(trial -> {
                  System.out.printf("Score: %.4f, Params: %s%n", 
                                  trial.getScore(), trial.getParameters());
              });
    }
}

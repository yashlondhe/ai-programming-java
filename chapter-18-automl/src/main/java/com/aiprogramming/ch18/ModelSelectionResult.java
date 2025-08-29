package com.aiprogramming.ch18;

import java.util.List;

/**
 * Result of model selection
 */
public class ModelSelectionResult {
    
    private final MLModel bestModel;
    private final double bestScore;
    private final List<ModelEvaluation> allEvaluations;
    
    public ModelSelectionResult(MLModel bestModel, double bestScore, List<ModelEvaluation> allEvaluations) {
        this.bestModel = bestModel;
        this.bestScore = bestScore;
        this.allEvaluations = allEvaluations;
    }
    
    public MLModel getBestModel() {
        return bestModel;
    }
    
    public double getBestScore() {
        return bestScore;
    }
    
    public List<ModelEvaluation> getAllEvaluations() {
        return allEvaluations;
    }
    
    /**
     * Print summary of model selection results
     */
    public void printSummary() {
        System.out.println("Best Model: " + bestModel.getClass().getSimpleName());
        System.out.println("Best Score: " + String.format("%.4f", bestScore));
        System.out.println("Total Models Evaluated: " + allEvaluations.size());
        
        System.out.println("\nAll Model Scores:");
        allEvaluations.stream()
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .forEach(eval -> {
                System.out.printf("%s: %.4f%n", 
                                eval.getModel().getClass().getSimpleName(), 
                                eval.getScore());
            });
    }
    
    /**
     * Helper class for model evaluation
     */
    public static class ModelEvaluation {
        private final MLModel model;
        private final double score;
        
        public ModelEvaluation(MLModel model, double score) {
            this.model = model;
            this.score = score;
        }
        
        public MLModel getModel() {
            return model;
        }
        
        public double getScore() {
            return score;
        }
    }
}

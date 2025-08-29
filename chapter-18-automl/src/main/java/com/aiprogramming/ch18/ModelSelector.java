package com.aiprogramming.ch18;

import java.util.*;

/**
 * Model selection using various evaluation metrics
 */
public class ModelSelector {
    
    private final ModelSelectionConfig config;
    
    public ModelSelector(ModelSelectionConfig config) {
        this.config = config;
    }
    
    /**
     * Select the best model from candidates
     */
    public ModelSelectionResult selectBestModel(double[][] features, double[] targets) {
        System.out.println("Starting model selection...");
        
        List<ModelEvaluation> evaluations = new ArrayList<>();
        
        // Evaluate each candidate model
        for (Class<? extends MLModel> modelClass : config.getCandidateModels()) {
            try {
                MLModel model = modelClass.getDeclaredConstructor().newInstance();
                double score = evaluateModel(model, features, targets);
                evaluations.add(new ModelEvaluation(model, score));
                
                System.out.println(modelClass.getSimpleName() + " score: " + String.format("%.4f", score));
            } catch (Exception e) {
                System.err.println("Error evaluating " + modelClass.getSimpleName() + ": " + e.getMessage());
            }
        }
        
        // Find best model
        ModelEvaluation bestEvaluation = evaluations.stream()
            .max((a, b) -> Double.compare(a.getScore(), b.getScore()))
            .orElse(null);
        
        if (bestEvaluation == null) {
            throw new RuntimeException("No models could be evaluated");
        }
        
        // Convert to ModelSelectionResult.ModelEvaluation
        List<ModelSelectionResult.ModelEvaluation> resultEvaluations = new ArrayList<>();
        for (ModelEvaluation eval : evaluations) {
            resultEvaluations.add(new ModelSelectionResult.ModelEvaluation(eval.getModel(), eval.getScore()));
        }
        
        return new ModelSelectionResult(bestEvaluation.getModel(), bestEvaluation.getScore(), resultEvaluations);
    }
    
    /**
     * Evaluate a model using cross-validation
     */
    private double evaluateModel(MLModel model, double[][] features, double[] targets) {
        if (config.isUseCrossValidation()) {
            return crossValidateModel(model, features, targets);
        } else {
            // Simple train/test split
            int splitIndex = (int) (features.length * 0.8);
            
            double[][] trainFeatures = Arrays.copyOfRange(features, 0, splitIndex);
            double[] trainTargets = Arrays.copyOfRange(targets, 0, splitIndex);
            double[][] testFeatures = Arrays.copyOfRange(features, splitIndex, features.length);
            double[] testTargets = Arrays.copyOfRange(targets, splitIndex, targets.length);
            
            model.train(trainFeatures, trainTargets);
            return calculateMetric(model, testFeatures, testTargets);
        }
    }
    
    /**
     * Cross-validate a model
     */
    private double crossValidateModel(MLModel model, double[][] features, double[] targets) {
        int folds = config.getCvFolds();
        double totalScore = 0.0;
        
        for (int fold = 0; fold < folds; fold++) {
            // Split data
            DataSplit split = splitData(features, targets, fold, folds);
            
            // Train and evaluate
            model.train(split.getTrainFeatures(), split.getTrainTargets());
            double score = calculateMetric(model, split.getTestFeatures(), split.getTestTargets());
            totalScore += score;
        }
        
        return totalScore / folds;
    }
    
    /**
     * Calculate the specified evaluation metric
     */
    private double calculateMetric(MLModel model, double[][] features, double[] targets) {
        double[] predictions = model.predict(features);
        
        switch (config.getEvaluationMetric()) {
            case ACCURACY:
                return calculateAccuracy(predictions, targets);
            case PRECISION:
                return calculatePrecision(predictions, targets);
            case RECALL:
                return calculateRecall(predictions, targets);
            case F1_SCORE:
                return calculateF1Score(predictions, targets);
            case ROC_AUC:
                return calculateROCAUC(predictions, targets);
            case MEAN_SQUARED_ERROR:
                return calculateMSE(predictions, targets);
            case MEAN_ABSOLUTE_ERROR:
                return calculateMAE(predictions, targets);
            case R2_SCORE:
                return calculateR2Score(predictions, targets);
            default:
                return calculateAccuracy(predictions, targets);
        }
    }
    
    /**
     * Split data for cross-validation
     */
    private DataSplit splitData(double[][] features, double[] targets, int fold, int totalFolds) {
        int foldSize = features.length / totalFolds;
        int startIndex = fold * foldSize;
        int endIndex = (fold == totalFolds - 1) ? features.length : (fold + 1) * foldSize;
        
        List<double[]> trainFeatures = new ArrayList<>();
        List<double[]> testFeatures = new ArrayList<>();
        List<Double> trainTargets = new ArrayList<>();
        List<Double> testTargets = new ArrayList<>();
        
        for (int i = 0; i < features.length; i++) {
            if (i >= startIndex && i < endIndex) {
                testFeatures.add(features[i]);
                testTargets.add(targets[i]);
            } else {
                trainFeatures.add(features[i]);
                trainTargets.add(targets[i]);
            }
        }
        
        return new DataSplit(
            trainFeatures.toArray(new double[0][]),
            trainTargets.stream().mapToDouble(Double::doubleValue).toArray(),
            testFeatures.toArray(new double[0][]),
            testTargets.stream().mapToDouble(Double::doubleValue).toArray()
        );
    }
    
    // Evaluation metric calculations
    
    private double calculateAccuracy(double[] predictions, double[] targets) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == Math.round(targets[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    private double calculatePrecision(double[] predictions, double[] targets) {
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == 1 && Math.round(targets[i]) == 1) {
                truePositives++;
            } else if (Math.round(predictions[i]) == 1 && Math.round(targets[i]) == 0) {
                falsePositives++;
            }
        }
        
        return truePositives + falsePositives == 0 ? 0 : (double) truePositives / (truePositives + falsePositives);
    }
    
    private double calculateRecall(double[] predictions, double[] targets) {
        int truePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == 1 && Math.round(targets[i]) == 1) {
                truePositives++;
            } else if (Math.round(predictions[i]) == 0 && Math.round(targets[i]) == 1) {
                falseNegatives++;
            }
        }
        
        return truePositives + falseNegatives == 0 ? 0 : (double) truePositives / (truePositives + falseNegatives);
    }
    
    private double calculateF1Score(double[] predictions, double[] targets) {
        double precision = calculatePrecision(predictions, targets);
        double recall = calculateRecall(predictions, targets);
        
        return precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    }
    
    private double calculateROCAUC(double[] predictions, double[] targets) {
        // Simplified ROC AUC calculation
        // In practice, this would use proper ROC curve calculation
        return calculateAccuracy(predictions, targets);
    }
    
    private double calculateMSE(double[] predictions, double[] targets) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.length;
    }
    
    private double calculateMAE(double[] predictions, double[] targets) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            sum += Math.abs(predictions[i] - targets[i]);
        }
        return sum / predictions.length;
    }
    
    private double calculateR2Score(double[] predictions, double[] targets) {
        double meanTarget = Arrays.stream(targets).average().orElse(0.0);
        
        double ssRes = 0.0;
        double ssTot = 0.0;
        
        for (int i = 0; i < predictions.length; i++) {
            ssRes += Math.pow(predictions[i] - targets[i], 2);
            ssTot += Math.pow(targets[i] - meanTarget, 2);
        }
        
        return ssTot == 0 ? 0 : 1 - (ssRes / ssTot);
    }
    
    /**
     * Helper class for model evaluation
     */
    private static class ModelEvaluation {
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

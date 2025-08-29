package com.aiprogramming.ch18;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;
import com.aiprogramming.utils.ValidationUtils;

/**
 * Hyperparameter optimization using various algorithms
 * Supports Bayesian optimization, grid search, and random search
 */
public class HyperparameterOptimizer {
    
    private final HyperparameterOptimizationConfig config;
    private final Random random;
    
    public HyperparameterOptimizer(HyperparameterOptimizationConfig config) {
        this.config = config;
        this.random = new Random(config.getRandomSeed());
    }
    
    /**
     * Optimize hyperparameters for a given model
     */
    public HyperparameterOptimizationResult optimize(MLModel model, double[][] features, double[] targets) {
        System.out.println("Starting hyperparameter optimization...");
        
        switch (config.getOptimizationMethod()) {
            case BAYESIAN:
                return bayesianOptimization(model, features, targets);
            case GRID_SEARCH:
                return gridSearch(model, features, targets);
            case RANDOM_SEARCH:
                return randomSearch(model, features, targets);
            default:
                return randomSearch(model, features, targets);
        }
    }
    
    /**
     * Bayesian optimization using Gaussian Process
     */
    private HyperparameterOptimizationResult bayesianOptimization(MLModel model, double[][] features, double[] targets) {
        System.out.println("Using Bayesian Optimization");
        
        List<HyperparameterTrial> trials = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        
        // Initial random trials
        int nInitial = Math.min(10, config.getMaxTrials());
        for (int i = 0; i < nInitial; i++) {
            Map<String, Object> params = generateRandomHyperparameters(model);
            double score = evaluateHyperparameters(model, params, features, targets);
            
            trials.add(new HyperparameterTrial(params, score));
            scores.add(score);
            
            System.out.printf("Trial %d: Score = %.4f%n", i + 1, score);
        }
        
        // Bayesian optimization iterations
        for (int i = nInitial; i < config.getMaxTrials(); i++) {
            // Find best parameters so far
            int bestIndex = findBestTrialIndex(trials);
            Map<String, Object> bestParams = trials.get(bestIndex).getParameters();
            
            // Generate next candidate using acquisition function
            Map<String, Object> nextParams = generateNextCandidate(model, trials, scores);
            double score = evaluateHyperparameters(model, nextParams, features, targets);
            
            trials.add(new HyperparameterTrial(nextParams, score));
            scores.add(score);
            
            System.out.printf("Trial %d: Score = %.4f%n", i + 1, score);
        }
        
        // Find best trial
        int bestIndex = findBestTrialIndex(trials);
        HyperparameterTrial bestTrial = trials.get(bestIndex);
        
        return new HyperparameterOptimizationResult(bestTrial.getParameters(), bestTrial.getScore(), trials);
    }
    
    /**
     * Grid search optimization
     */
    private HyperparameterOptimizationResult gridSearch(MLModel model, double[][] features, double[] targets) {
        System.out.println("Using Grid Search");
        
        List<HyperparameterTrial> trials = new ArrayList<>();
        List<Map<String, Object>> parameterGrid = generateParameterGrid(model);
        
        for (Map<String, Object> params : parameterGrid) {
            double score = evaluateHyperparameters(model, params, features, targets);
            trials.add(new HyperparameterTrial(params, score));
        }
        
        int bestIndex = findBestTrialIndex(trials);
        HyperparameterTrial bestTrial = trials.get(bestIndex);
        
        return new HyperparameterOptimizationResult(bestTrial.getParameters(), bestTrial.getScore(), trials);
    }
    
    /**
     * Random search optimization
     */
    private HyperparameterOptimizationResult randomSearch(MLModel model, double[][] features, double[] targets) {
        System.out.println("Using Random Search");
        
        List<HyperparameterTrial> trials = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;
        Map<String, Object> bestParams = null;
        
        for (int i = 0; i < config.getMaxTrials(); i++) {
            Map<String, Object> params = generateRandomHyperparameters(model);
            double score = evaluateHyperparameters(model, params, features, targets);
            
            trials.add(new HyperparameterTrial(params, score));
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
            }
            
            System.out.printf("Trial %d: Score = %.4f%n", i + 1, score);
        }
        
        return new HyperparameterOptimizationResult(bestParams, bestScore, trials);
    }
    
    /**
     * Generate random hyperparameters for a model
     */
    private Map<String, Object> generateRandomHyperparameters(MLModel model) {
        Map<String, Object> params = new HashMap<>();
        
        if (model instanceof LinearRegression) {
            params.put("learningRate", random.nextDouble() * 0.1 + 0.001);
            params.put("maxIterations", random.nextInt(1000) + 100);
        } else if (model instanceof LogisticRegression) {
            params.put("learningRate", random.nextDouble() * 0.1 + 0.001);
            params.put("maxIterations", random.nextInt(1000) + 100);
            params.put("regularization", random.nextDouble() * 0.1);
        } else if (model instanceof RandomForest) {
            params.put("numTrees", random.nextInt(50) + 10);
            params.put("maxDepth", random.nextInt(10) + 3);
            params.put("minSamplesSplit", random.nextInt(10) + 2);
        } else if (model instanceof NeuralNetwork) {
            params.put("learningRate", random.nextDouble() * 0.1 + 0.001);
            params.put("hiddenLayers", random.nextInt(3) + 1);
            params.put("neuronsPerLayer", random.nextInt(50) + 10);
            params.put("dropout", random.nextDouble() * 0.5);
        }
        
        return params;
    }
    
    /**
     * Generate parameter grid for grid search
     */
    private List<Map<String, Object>> generateParameterGrid(MLModel model) {
        List<Map<String, Object>> grid = new ArrayList<>();
        
        if (model instanceof LinearRegression) {
            double[] learningRates = {0.001, 0.01, 0.1};
            int[] maxIterations = {100, 500, 1000};
            
            for (double lr : learningRates) {
                for (int maxIter : maxIterations) {
                    Map<String, Object> params = new HashMap<>();
                    params.put("learningRate", lr);
                    params.put("maxIterations", maxIter);
                    grid.add(params);
                }
            }
        } else if (model instanceof RandomForest) {
            int[] numTrees = {10, 25, 50};
            int[] maxDepths = {3, 5, 7};
            
            for (int trees : numTrees) {
                for (int depth : maxDepths) {
                    Map<String, Object> params = new HashMap<>();
                    params.put("numTrees", trees);
                    params.put("maxDepth", depth);
                    params.put("minSamplesSplit", 2);
                    grid.add(params);
                }
            }
        }
        
        return grid;
    }
    
    /**
     * Evaluate hyperparameters using cross-validation
     */
    private double evaluateHyperparameters(MLModel model, Map<String, Object> params, 
                                         double[][] features, double[] targets) {
        // Set hyperparameters
        model.setHyperparameters(params);
        
        // Cross-validation
        int folds = config.getCvFolds();
        double totalScore = 0.0;
        
        for (int fold = 0; fold < folds; fold++) {
            // Split data
            DataSplit split = splitData(features, targets, fold, folds);
            
            // Train and evaluate
            model.train(split.getTrainFeatures(), split.getTrainTargets());
            double score = model.evaluate(split.getTestFeatures(), split.getTestTargets());
            totalScore += score;
        }
        
        return totalScore / folds;
    }
    
    /**
     * Generate next candidate for Bayesian optimization
     */
    private Map<String, Object> generateNextCandidate(MLModel model, List<HyperparameterTrial> trials, 
                                                    List<Double> scores) {
        // Simple approach: choose random parameters but bias towards better regions
        // In a real implementation, this would use Gaussian Process regression
        
        // For now, use a simple strategy: 70% random, 30% near best
        if (random.nextDouble() < 0.3) {
            // Generate parameters near the best trial
            int bestIndex = findBestTrialIndex(trials);
            Map<String, Object> bestParams = trials.get(bestIndex).getParameters();
            return generateNearbyHyperparameters(model, bestParams);
        } else {
            return generateRandomHyperparameters(model);
        }
    }
    
    /**
     * Generate hyperparameters near given parameters
     */
    private Map<String, Object> generateNearbyHyperparameters(MLModel model, Map<String, Object> baseParams) {
        Map<String, Object> params = new HashMap<>();
        
        for (Map.Entry<String, Object> entry : baseParams.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            
            if (value instanceof Double) {
                double baseValue = (Double) value;
                double noise = baseValue * 0.1 * (random.nextDouble() - 0.5);
                params.put(key, baseValue + noise);
            } else if (value instanceof Integer) {
                int baseValue = (Integer) value;
                int noise = (int) (baseValue * 0.1 * (random.nextDouble() - 0.5));
                params.put(key, baseValue + noise);
            } else {
                params.put(key, value);
            }
        }
        
        return params;
    }
    
    /**
     * Find index of best trial
     */
    private int findBestTrialIndex(List<HyperparameterTrial> trials) {
        int bestIndex = 0;
        double bestScore = trials.get(0).getScore();
        
        for (int i = 1; i < trials.size(); i++) {
            if (trials.get(i).getScore() > bestScore) {
                bestScore = trials.get(i).getScore();
                bestIndex = i;
            }
        }
        
        return bestIndex;
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
}

# Chapter 18: AutoML and Neural Architecture Search

## Introduction

Automated Machine Learning (AutoML) and Neural Architecture Search (NAS) represent the cutting edge of machine learning automation. These technologies aim to reduce the manual effort required in the machine learning pipeline by automatically handling feature selection, model selection, hyperparameter optimization, and even neural network architecture design. This chapter explores the implementation of AutoML and NAS systems in Java, providing a comprehensive framework for automated machine learning.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of AutoML and its components
- Implement automated hyperparameter optimization using various algorithms
- Design and implement feature selection algorithms
- Build automated model selection systems
- Understand and implement Neural Architecture Search
- Create complete AutoML pipelines
- Evaluate and compare different AutoML approaches
- Apply AutoML techniques to real-world problems

### Key Concepts

- **AutoML**: Automated machine learning pipeline that handles feature selection, model selection, and hyperparameter optimization
- **Hyperparameter Optimization**: Automated tuning of model parameters using algorithms like Bayesian optimization, grid search, and random search
- **Feature Selection**: Automated identification of the most relevant features for a given problem
- **Model Selection**: Automated comparison and selection of the best machine learning model
- **Neural Architecture Search (NAS)**: Automated discovery of optimal neural network architectures
- **Cross-validation**: Robust evaluation technique for model performance assessment
- **Time Budgeting**: Managing computational resources in AutoML systems

## 18.1 Automated Machine Learning (AutoML)

AutoML represents a paradigm shift in machine learning, automating the traditionally manual and time-consuming process of building effective machine learning models.

### 18.1.1 AutoML Pipeline Components

A complete AutoML pipeline consists of several key components:

1. **Feature Selection**: Identifying the most relevant features
2. **Model Selection**: Choosing the best algorithm for the problem
3. **Hyperparameter Optimization**: Tuning model parameters
4. **Model Evaluation**: Assessing performance using cross-validation

#### AutoML Framework

```java
package com.aiprogramming.ch18;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

/**
 * Automated Machine Learning (AutoML) framework
 * Provides automated hyperparameter tuning, feature selection, and model selection
 */
public class AutoML {
    
    private final HyperparameterOptimizer hyperparameterOptimizer;
    private final FeatureSelector featureSelector;
    private final ModelSelector modelSelector;
    private final AutoMLConfig config;
    
    public AutoML(AutoMLConfig config) {
        this.config = config;
        this.hyperparameterOptimizer = new HyperparameterOptimizer(config.getOptimizationConfig());
        this.featureSelector = new FeatureSelector(config.getFeatureSelectionConfig());
        this.modelSelector = new ModelSelector(config.getModelSelectionConfig());
    }
    
    /**
     * Run complete AutoML pipeline
     */
    public AutoMLResult runAutoML(double[][] features, double[] targets) {
        System.out.println("Starting AutoML pipeline...");
        
        // Step 1: Feature Selection
        System.out.println("Step 1: Feature Selection");
        FeatureSelectionResult featureResult = featureSelector.selectFeatures(features, targets);
        
        // Step 2: Model Selection
        System.out.println("Step 2: Model Selection");
        ModelSelectionResult modelResult = modelSelector.selectBestModel(
            featureResult.getSelectedFeatures(), targets);
        
        // Step 3: Hyperparameter Optimization
        System.out.println("Step 3: Hyperparameter Optimization");
        HyperparameterOptimizationResult hpResult = hyperparameterOptimizer.optimize(
            modelResult.getBestModel(), 
            featureResult.getSelectedFeatures(), 
            targets);
        
        return new AutoMLResult(featureResult, modelResult, hpResult);
    }
    
    /**
     * Run hyperparameter optimization only
     */
    public HyperparameterOptimizationResult optimizeHyperparameters(
            MLModel model, double[][] features, double[] targets) {
        return hyperparameterOptimizer.optimize(model, features, targets);
    }
    
    /**
     * Run feature selection only
     */
    public FeatureSelectionResult selectFeatures(double[][] features, double[] targets) {
        return featureSelector.selectFeatures(features, targets);
    }
    
    /**
     * Run model selection only
     */
    public ModelSelectionResult selectBestModel(double[][] features, double[] targets) {
        return modelSelector.selectBestModel(features, targets);
    }
}
```

### 18.1.2 Configuration Management

The AutoML framework uses a comprehensive configuration system to manage all aspects of the pipeline:

```java
package com.aiprogramming.ch18;

/**
 * Configuration class for AutoML pipeline
 */
public class AutoMLConfig {
    
    private final HyperparameterOptimizationConfig optimizationConfig;
    private final FeatureSelectionConfig featureSelectionConfig;
    private final ModelSelectionConfig modelSelectionConfig;
    private final int maxTimeSeconds;
    private final int maxTrials;
    private final int cvFolds;
    
    public AutoMLConfig(Builder builder) {
        this.optimizationConfig = builder.optimizationConfig;
        this.featureSelectionConfig = builder.featureSelectionConfig;
        this.modelSelectionConfig = builder.modelSelectionConfig;
        this.maxTimeSeconds = builder.maxTimeSeconds;
        this.maxTrials = builder.maxTrials;
        this.cvFolds = builder.cvFolds;
    }
    
    public static class Builder {
        private HyperparameterOptimizationConfig optimizationConfig = new HyperparameterOptimizationConfig();
        private FeatureSelectionConfig featureSelectionConfig = new FeatureSelectionConfig();
        private ModelSelectionConfig modelSelectionConfig = new ModelSelectionConfig();
        private int maxTimeSeconds = 3600; // 1 hour default
        private int maxTrials = 100;
        private int cvFolds = 5;
        
        public Builder maxTrials(int maxTrials) {
            this.maxTrials = maxTrials;
            return this;
        }
        
        public Builder cvFolds(int cvFolds) {
            this.cvFolds = cvFolds;
            return this;
        }
        
        public AutoMLConfig build() {
            return new AutoMLConfig(this);
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
}
```

## 18.2 Hyperparameter Optimization

Hyperparameter optimization is a critical component of AutoML that automatically finds the best parameters for machine learning models.

### 18.2.1 Optimization Algorithms

The framework supports multiple optimization strategies:

#### Random Search

```java
package com.aiprogramming.ch18;

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
```

#### Grid Search

```java
package com.aiprogramming.ch18;

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
```

#### Bayesian Optimization

```java
package com.aiprogramming.ch18;

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
```

### 18.2.2 Hyperparameter Generation

The framework includes intelligent hyperparameter generation for different model types:

```java
package com.aiprogramming.ch18;

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
```

## 18.3 Feature Selection

Feature selection is crucial for reducing dimensionality and improving model performance.

### 18.3.1 Feature Selection Methods

The framework implements multiple feature selection algorithms:

#### Correlation-based Selection

```java
package com.aiprogramming.ch18;

/**
 * Select features based on correlation with target
 */
private FeatureSelectionResult selectByCorrelation(double[][] features, double[] targets) {
    int numFeatures = features[0].length;
    double[] correlations = new double[numFeatures];
    
    // Calculate correlation for each feature
    for (int i = 0; i < numFeatures; i++) {
        double[] featureValues = new double[features.length];
        for (int j = 0; j < features.length; j++) {
            featureValues[j] = features[j][i];
        }
        correlations[i] = Math.abs(calculateCorrelation(featureValues, targets));
    }
    
    // Select top features
    return selectTopFeatures(correlations, features);
}

/**
 * Calculate correlation between two arrays
 */
private double calculateCorrelation(double[] x, double[] y) {
    double meanX = Arrays.stream(x).average().orElse(0.0);
    double meanY = Arrays.stream(y).average().orElse(0.0);
    
    double numerator = 0.0;
    double sumXSquared = 0.0;
    double sumYSquared = 0.0;
    
    for (int i = 0; i < x.length; i++) {
        double xDiff = x[i] - meanX;
        double yDiff = y[i] - meanY;
        numerator += xDiff * yDiff;
        sumXSquared += xDiff * xDiff;
        sumYSquared += yDiff * yDiff;
    }
    
    double denominator = Math.sqrt(sumXSquared * sumYSquared);
    return denominator == 0 ? 0 : numerator / denominator;
}
```

#### Recursive Feature Elimination

```java
package com.aiprogramming.ch18;

/**
 * Select features using recursive feature elimination
 */
private FeatureSelectionResult selectByRecursiveFeatureElimination(double[][] features, double[] targets) {
    int numFeatures = features[0].length;
    List<Integer> remainingFeatures = new ArrayList<>();
    for (int i = 0; i < numFeatures; i++) {
        remainingFeatures.add(i);
    }
    
    while (remainingFeatures.size() > config.getMaxFeatures()) {
        // Train model with remaining features
        double[][] currentFeatures = selectFeatures(features, remainingFeatures);
        LinearRegression model = new LinearRegression();
        model.train(currentFeatures, targets);
        
        // Find feature with lowest importance
        double[] coefficients = model.getCoefficients();
        int worstFeatureIndex = 0;
        double minImportance = Math.abs(coefficients[0]);
        
        for (int i = 1; i < coefficients.length; i++) {
            if (Math.abs(coefficients[i]) < minImportance) {
                minImportance = Math.abs(coefficients[i]);
                worstFeatureIndex = i;
            }
        }
        
        // Remove worst feature
        remainingFeatures.remove(worstFeatureIndex);
    }
    
    int[] selectedIndices = remainingFeatures.stream().mapToInt(Integer::intValue).toArray();
    double[][] selectedFeatures = selectFeatures(features, selectedIndices);
    
    return new FeatureSelectionResult(selectedIndices, selectedFeatures, 
                                    new double[selectedIndices.length]);
}
```

### 18.3.2 Feature Selection Configuration

```java
package com.aiprogramming.ch18;

/**
 * Configuration for feature selection
 */
public class FeatureSelectionConfig {
    
    public enum SelectionMethod {
        CORRELATION,
        MUTUAL_INFORMATION,
        RECURSIVE_FEATURE_ELIMINATION,
        LASSO,
        RANDOM_FOREST_IMPORTANCE
    }
    
    private final SelectionMethod selectionMethod;
    private final int maxFeatures;
    private final double threshold;
    private final boolean useCrossValidation;
    
    public FeatureSelectionConfig() {
        this(SelectionMethod.CORRELATION, 10, 0.01, true);
    }
    
    public FeatureSelectionConfig(SelectionMethod selectionMethod, int maxFeatures, 
                                double threshold, boolean useCrossValidation) {
        this.selectionMethod = selectionMethod;
        this.maxFeatures = maxFeatures;
        this.threshold = threshold;
        this.useCrossValidation = useCrossValidation;
    }
}
```

## 18.4 Model Selection

Model selection automatically compares different algorithms and selects the best one for a given problem.

### 18.4.1 Model Selection Framework

```java
package com.aiprogramming.ch18;

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
}
```

### 18.4.2 Evaluation Metrics

The framework supports multiple evaluation metrics:

```java
package com.aiprogramming.ch18;

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

private double calculateMSE(double[] predictions, double[] targets) {
    double sum = 0.0;
    for (int i = 0; i < predictions.length; i++) {
        double error = predictions[i] - targets[i];
        sum += error * error;
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
```

## 18.5 Neural Architecture Search (NAS)

Neural Architecture Search automates the design of neural network architectures.

### 18.5.1 NAS Framework

```java
package com.aiprogramming.ch18;

/**
 * Neural Architecture Search (NAS) implementation
 * Automatically discovers optimal neural network architectures
 */
public class NeuralArchitectureSearch {
    
    private final NASConfig config;
    private final Random random;
    
    public NeuralArchitectureSearch(NASConfig config) {
        this.config = config;
        this.random = new Random(config.getRandomSeed());
    }
    
    /**
     * Search for optimal neural network architecture
     */
    public NASResult search(double[][] features, double[] targets) {
        System.out.println("Starting Neural Architecture Search...");
        
        List<ArchitectureTrial> trials = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;
        NeuralArchitecture bestArchitecture = null;
        
        for (int trial = 0; trial < config.getMaxTrials(); trial++) {
            System.out.printf("Trial %d/%d%n", trial + 1, config.getMaxTrials());
            
            // Generate random architecture
            NeuralArchitecture architecture = generateRandomArchitecture();
            
            // Train and evaluate architecture
            double score = evaluateArchitecture(architecture, features, targets);
            
            trials.add(new ArchitectureTrial(architecture, score));
            
            if (score > bestScore) {
                bestScore = score;
                bestArchitecture = architecture;
                System.out.printf("New best architecture found! Score: %.4f%n", score);
            }
        }
        
        return new NASResult(bestArchitecture, bestScore, trials);
    }
}
```

### 18.5.2 Architecture Generation

```java
package com.aiprogramming.ch18;

/**
 * Generate a random neural network architecture
 */
private NeuralArchitecture generateRandomArchitecture() {
    int numLayers = random.nextInt(config.getMaxLayers() - config.getMinLayers() + 1) + config.getMinLayers();
    List<LayerConfig> layers = new ArrayList<>();
    
    int currentSize = config.getInputSize();
    
    for (int i = 0; i < numLayers; i++) {
        // Random layer size
        int layerSize = random.nextInt(config.getMaxLayerSize() - config.getMinLayerSize() + 1) + config.getMinLayerSize();
        
        // Random activation function
        String activation = config.getActivationFunctions()[random.nextInt(config.getActivationFunctions().length)];
        
        // Random dropout rate
        double dropout = random.nextDouble() * config.getMaxDropout();
        
        layers.add(new LayerConfig(layerSize, activation, dropout));
        currentSize = layerSize;
    }
    
    return new NeuralArchitecture(layers, config.getOutputSize());
}

/**
 * Neural network architecture specification
 */
public static class NeuralArchitecture {
    private final List<LayerConfig> layers;
    private final int outputSize;
    
    public NeuralArchitecture(List<LayerConfig> layers, int outputSize) {
        this.layers = layers;
        this.outputSize = outputSize;
    }
    
    public List<LayerConfig> getLayers() {
        return layers;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NeuralArchitecture{layers=[");
        for (int i = 0; i < layers.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(layers.get(i));
        }
        sb.append("], outputSize=").append(outputSize).append("}");
        return sb.toString();
    }
}
```

### 18.5.3 NAS Configuration

```java
package com.aiprogramming.ch18;

/**
 * Configuration for Neural Architecture Search
 */
public class NASConfig {
    
    private final int maxTrials;
    private final int minLayers;
    private final int maxLayers;
    private final int minLayerSize;
    private final int maxLayerSize;
    private final int inputSize;
    private final int outputSize;
    private final double maxDropout;
    private final String[] activationFunctions;
    private final long randomSeed;
    
    public NASConfig() {
        this(50, 1, 5, 10, 100, 10, 1, 0.5, 
             new String[]{"relu", "tanh", "sigmoid"}, 42L);
    }
    
    public NASConfig(int maxTrials, int minLayers, int maxLayers, int minLayerSize, 
                    int maxLayerSize, int inputSize, int outputSize, double maxDropout,
                    String[] activationFunctions, long randomSeed) {
        this.maxTrials = maxTrials;
        this.minLayers = minLayers;
        this.maxLayers = maxLayers;
        this.minLayerSize = minLayerSize;
        this.maxLayerSize = maxLayerSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.maxDropout = maxDropout;
        this.activationFunctions = activationFunctions;
        this.randomSeed = randomSeed;
    }
}
```

## 18.6 Machine Learning Models

The framework includes several machine learning model implementations.

### 18.6.1 Model Interface

```java
package com.aiprogramming.ch18;

import java.util.Map;

/**
 * Interface for machine learning models
 */
public interface MLModel {
    
    /**
     * Train the model on the given data
     */
    void train(double[][] features, double[] targets);
    
    /**
     * Make predictions on new data
     */
    double[] predict(double[][] features);
    
    /**
     * Evaluate the model on test data
     */
    double evaluate(double[][] features, double[] targets);
    
    /**
     * Set hyperparameters for the model
     */
    void setHyperparameters(Map<String, Object> hyperparameters);
    
    /**
     * Get the current hyperparameters
     */
    Map<String, Object> getHyperparameters();
}
```

### 18.6.2 Linear Regression

```java
package com.aiprogramming.ch18;

/**
 * Simple Linear Regression model
 */
public class LinearRegression implements MLModel {
    
    private double[] coefficients;
    private double intercept;
    private Map<String, Object> hyperparameters;
    
    public LinearRegression() {
        this.hyperparameters = new HashMap<>();
        this.hyperparameters.put("learningRate", 0.01);
        this.hyperparameters.put("maxIterations", 1000);
        this.hyperparameters.put("regularization", 0.0);
    }
    
    @Override
    public void train(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        int numSamples = features.length;
        
        // Initialize coefficients
        coefficients = new double[numFeatures];
        intercept = 0.0;
        
        double learningRate = (Double) hyperparameters.get("learningRate");
        int maxIterations = (Integer) hyperparameters.get("maxIterations");
        double regularization = (Double) hyperparameters.get("regularization");
        
        // Gradient descent
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[] gradients = new double[numFeatures];
            double interceptGradient = 0.0;
            
            // Calculate gradients
            for (int i = 0; i < numSamples; i++) {
                double prediction = predict(features[i]);
                double error = prediction - targets[i];
                
                // Feature gradients
                for (int j = 0; j < numFeatures; j++) {
                    gradients[j] += error * features[i][j];
                }
                
                // Intercept gradient
                interceptGradient += error;
            }
            
            // Update parameters
            for (int j = 0; j < numFeatures; j++) {
                double gradient = gradients[j] / numSamples;
                if (regularization > 0) {
                    gradient += regularization * coefficients[j];
                }
                coefficients[j] -= learningRate * gradient;
            }
            
            intercept -= learningRate * (interceptGradient / numSamples);
        }
    }
    
    private double predict(double[] features) {
        double prediction = intercept;
        for (int i = 0; i < features.length; i++) {
            prediction += coefficients[i] * features[i];
        }
        return prediction;
    }
    
    @Override
    public double[] predict(double[][] features) {
        double[] predictions = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            predictions[i] = predict(features[i]);
        }
        return predictions;
    }
    
    @Override
    public double evaluate(double[][] features, double[] targets) {
        double[] predictions = predict(features);
        double mse = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - targets[i];
            mse += error * error;
        }
        return mse / predictions.length;
    }
    
    public double[] getCoefficients() {
        return coefficients;
    }
    
    public double getIntercept() {
        return intercept;
    }
}
```

### 18.6.3 Random Forest

```java
package com.aiprogramming.ch18;

/**
 * Simple Random Forest model
 */
public class RandomForest implements MLModel {
    
    private List<DecisionTree> trees;
    private Map<String, Object> hyperparameters;
    private double[] featureImportance;
    
    public RandomForest() {
        this.hyperparameters = new HashMap<>();
        this.hyperparameters.put("numTrees", 10);
        this.hyperparameters.put("maxDepth", 5);
        this.hyperparameters.put("minSamplesSplit", 2);
        this.trees = new ArrayList<>();
    }
    
    @Override
    public void train(double[][] features, double[] targets) {
        int numTrees = (Integer) hyperparameters.get("numTrees");
        int maxDepth = (Integer) hyperparameters.get("maxDepth");
        int minSamplesSplit = (Integer) hyperparameters.get("minSamplesSplit");
        
        trees.clear();
        
        // Train multiple trees
        for (int i = 0; i < numTrees; i++) {
            DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit);
            
            // Bootstrap sample
            int[] bootstrapIndices = generateBootstrapSample(features.length);
            double[][] bootstrapFeatures = new double[bootstrapIndices.length][features[0].length];
            double[] bootstrapTargets = new double[bootstrapIndices.length];
            
            for (int j = 0; j < bootstrapIndices.length; j++) {
                bootstrapFeatures[j] = features[bootstrapIndices[j]];
                bootstrapTargets[j] = targets[bootstrapIndices[j]];
            }
            
            tree.train(bootstrapFeatures, bootstrapTargets);
            trees.add(tree);
        }
        
        // Calculate feature importance
        calculateFeatureImportance(features, targets);
    }
    
    @Override
    public double[] predict(double[][] features) {
        double[] predictions = new double[features.length];
        
        for (int i = 0; i < features.length; i++) {
            double sum = 0.0;
            for (DecisionTree tree : trees) {
                sum += tree.predict(features[i]);
            }
            predictions[i] = sum / trees.size();
        }
        
        return predictions;
    }
    
    public double[] getFeatureImportance() {
        return featureImportance;
    }
}
```

## 18.7 Practical Example

### 18.7.1 Complete AutoML Pipeline

A comprehensive example demonstrating the complete AutoML pipeline:

```java
package com.aiprogramming.ch18;

/**
 * Comprehensive demonstration of AutoML and Neural Architecture Search
 */
public class AutoMLDemo {
    
    public static void main(String[] args) {
        System.out.println("=== AutoML and Neural Architecture Search Demo ===\n");
        
        // Generate sample data
        double[][] features = generateSampleData(1000, 20);
        double[] targets = generateTargets(features);
        
        System.out.println("Generated dataset with " + features.length + " samples and " + 
                          features[0].length + " features");
        
        // Demonstrate AutoML pipeline
        demonstrateAutoML(features, targets);
        
        // Demonstrate Neural Architecture Search
        demonstrateNAS(features, targets);
        
        // Demonstrate individual components
        demonstrateFeatureSelection(features, targets);
        demonstrateModelSelection(features, targets);
        demonstrateHyperparameterOptimization(features, targets);
        
        System.out.println("\n=== Demo Complete ===");
    }
    
    /**
     * Demonstrate complete AutoML pipeline
     */
    private static void demonstrateAutoML(double[][] features, double[] targets) {
        System.out.println("\n=== Complete AutoML Pipeline ===");
        
        // Configure AutoML
        AutoMLConfig config = AutoMLConfig.builder()
            .maxTrials(20)
            .cvFolds(3)
            .build();
        
        AutoML autoML = new AutoML(config);
        
        // Run complete pipeline
        AutoMLResult result = autoML.runAutoML(features, targets);
        
        // Print results
        result.printSummary();
    }
    
    /**
     * Demonstrate Neural Architecture Search
     */
    private static void demonstrateNAS(double[][] features, double[] targets) {
        System.out.println("\n=== Neural Architecture Search ===");
        
        // Configure NAS
        NASConfig nasConfig = new NASConfig(10, 1, 3, 5, 50, features[0].length, 1, 0.3,
                                           new String[]{"relu", "tanh"}, 42L);
        
        NeuralArchitectureSearch nas = new NeuralArchitectureSearch(nasConfig);
        
        // Run NAS
        NeuralArchitectureSearch.NASResult result = nas.search(features, targets);
        
        // Print results
        result.printSummary();
    }
}
```

### 18.7.2 Expected Results

The AutoML pipeline typically produces:

1. **Feature Selection**: Identifies the most relevant features
2. **Model Selection**: Chooses the best performing algorithm
3. **Hyperparameter Optimization**: Finds optimal parameters
4. **Final Model**: Optimized model ready for deployment

## 18.8 Advanced Topics

### 18.8.1 Time Budgeting

Managing computational resources in AutoML:

```java
package com.aiprogramming.ch18;

/**
 * Time-budgeted AutoML configuration
 */
public class TimeBudgetedAutoML {
    
    public static AutoMLResult runWithTimeBudget(double[][] features, double[] targets, 
                                               int maxTimeSeconds) {
        long startTime = System.currentTimeMillis();
        
        AutoMLConfig config = AutoMLConfig.builder()
            .maxTimeSeconds(maxTimeSeconds)
            .maxTrials(1000) // High number, will be limited by time
            .build();
        
        AutoML autoML = new AutoML(config);
        
        // Run AutoML with time monitoring
        AutoMLResult result = autoML.runAutoML(features, targets);
        
        long endTime = System.currentTimeMillis();
        System.out.printf("AutoML completed in %.2f seconds%n", 
                         (endTime - startTime) / 1000.0);
        
        return result;
    }
}
```

### 18.8.2 Parallel Optimization

Implementing parallel hyperparameter optimization:

```java
package com.aiprogramming.ch18;

import java.util.concurrent.*;

/**
 * Parallel hyperparameter optimization
 */
public class ParallelHyperparameterOptimizer {
    
    private final ExecutorService executor;
    private final int numThreads;
    
    public ParallelHyperparameterOptimizer(int numThreads) {
        this.numThreads = numThreads;
        this.executor = Executors.newFixedThreadPool(numThreads);
    }
    
    public List<HyperparameterTrial> optimizeParallel(MLModel model, double[][] features, 
                                                     double[] targets, int numTrials) {
        List<Future<HyperparameterTrial>> futures = new ArrayList<>();
        
        // Submit trials to thread pool
        for (int i = 0; i < numTrials; i++) {
            Future<HyperparameterTrial> future = executor.submit(() -> {
                Map<String, Object> params = generateRandomHyperparameters(model);
                double score = evaluateHyperparameters(model, params, features, targets);
                return new HyperparameterTrial(params, score);
            });
            futures.add(future);
        }
        
        // Collect results
        List<HyperparameterTrial> trials = new ArrayList<>();
        for (Future<HyperparameterTrial> future : futures) {
            try {
                trials.add(future.get());
            } catch (Exception e) {
                System.err.println("Error in parallel optimization: " + e.getMessage());
            }
        }
        
        return trials;
    }
    
    public void shutdown() {
        executor.shutdown();
    }
}
```

### 18.8.3 Multi-Objective Optimization

Optimizing multiple objectives simultaneously:

```java
package com.aiprogramming.ch18;

/**
 * Multi-objective optimization for AutoML
 */
public class MultiObjectiveAutoML {
    
    public static class Objective {
        public enum Type { ACCURACY, INTERPRETABILITY, EFFICIENCY }
        
        private final Type type;
        private final double weight;
        
        public Objective(Type type, double weight) {
            this.type = type;
            this.weight = weight;
        }
    }
    
    public static AutoMLResult optimizeMultiObjective(double[][] features, double[] targets,
                                                    List<Objective> objectives) {
        // Implement multi-objective optimization
        // This could use techniques like Pareto optimization or weighted sum
        
        AutoMLConfig config = AutoMLConfig.builder()
            .maxTrials(50)
            .build();
        
        AutoML autoML = new AutoML(config);
        AutoMLResult result = autoML.runAutoML(features, targets);
        
        // Apply multi-objective scoring
        double multiObjectiveScore = calculateMultiObjectiveScore(result, objectives);
        System.out.println("Multi-objective score: " + multiObjectiveScore);
        
        return result;
    }
    
    private static double calculateMultiObjectiveScore(AutoMLResult result, List<Objective> objectives) {
        double totalScore = 0.0;
        
        for (Objective objective : objectives) {
            double score = 0.0;
            
            switch (objective.type) {
                case ACCURACY:
                    score = result.getBestScore();
                    break;
                case INTERPRETABILITY:
                    // Simple interpretability score based on model type
                    String modelType = result.getFinalModel().getClass().getSimpleName();
                    if (modelType.equals("LinearRegression")) {
                        score = 1.0;
                    } else if (modelType.equals("RandomForest")) {
                        score = 0.8;
                    } else {
                        score = 0.3;
                    }
                    break;
                case EFFICIENCY:
                    // Efficiency score based on model complexity
                    score = 1.0 / (1.0 + result.getFeatureSelectionResult().getSelectedFeatureIndices().length);
                    break;
            }
            
            totalScore += objective.weight * score;
        }
        
        return totalScore;
    }
}
```

## 18.9 Best Practices

### 18.9.1 Data Preparation

- **Normalization**: Scale features appropriately before AutoML
- **Missing Values**: Handle missing data before feature selection
- **Outliers**: Detect and handle outliers that could affect optimization
- **Data Quality**: Ensure high-quality training data

### 18.9.2 Configuration

- **Start Simple**: Begin with basic configurations and gradually increase complexity
- **Domain Knowledge**: Use domain expertise to guide feature selection and model choices
- **Resource Constraints**: Set appropriate time and computational budgets
- **Validation Strategy**: Use proper cross-validation to avoid overfitting

### 18.9.3 Evaluation

- **Multiple Metrics**: Evaluate models using multiple performance metrics
- **Holdout Set**: Maintain a separate test set for final evaluation
- **Statistical Significance**: Test for significant differences between models
- **Business Impact**: Consider business requirements in model selection

### 18.9.4 Production Deployment

- **Model Persistence**: Save optimized models and configurations
- **Versioning**: Implement model versioning for tracking changes
- **Monitoring**: Set up monitoring for model performance in production
- **Retraining**: Plan for periodic model retraining

## 18.10 Summary

In this chapter, we explored the implementation of AutoML and Neural Architecture Search in Java:

1. **AutoML Framework**: Complete automated machine learning pipeline
2. **Hyperparameter Optimization**: Multiple optimization algorithms (random search, grid search, Bayesian optimization)
3. **Feature Selection**: Various feature selection methods (correlation, mutual information, recursive elimination)
4. **Model Selection**: Automated model comparison and selection
5. **Neural Architecture Search**: Automated neural network architecture discovery
6. **Machine Learning Models**: Implementations of common algorithms
7. **Advanced Topics**: Time budgeting, parallel optimization, multi-objective optimization

### Key Takeaways

- **AutoML** automates the machine learning pipeline, reducing manual effort
- **Hyperparameter Optimization** is crucial for model performance
- **Feature Selection** improves model interpretability and performance
- **Model Selection** ensures the best algorithm is chosen for the problem
- **Neural Architecture Search** automates neural network design
- **Proper Configuration** is essential for effective AutoML
- **Evaluation and Monitoring** are critical for production deployment

### Next Steps

- Explore more sophisticated optimization algorithms
- Implement ensemble methods in AutoML
- Study advanced NAS techniques (evolutionary algorithms, reinforcement learning)
- Apply AutoML to specific domains (computer vision, NLP, time series)
- Investigate AutoML for edge devices and mobile applications
- Research interpretable AutoML systems

## Exercises

### Exercise 1: Custom Feature Selection
Implement a new feature selection method (e.g., chi-square test, ANOVA) and integrate it into the AutoML framework.

### Exercise 2: Advanced Hyperparameter Optimization
Implement more sophisticated optimization algorithms like Tree-structured Parzen Estimators (TPE) or Hyperband.

### Exercise 3: Multi-Objective AutoML
Extend the AutoML framework to handle multiple objectives (accuracy, interpretability, efficiency) simultaneously.

### Exercise 4: Neural Architecture Search
Implement evolutionary algorithms for neural architecture search and compare with random search.

### Exercise 5: Time-Budgeted AutoML
Create a time-budgeted AutoML system that adapts its search strategy based on remaining time.

### Exercise 6: AutoML for Specific Domains
Adapt the AutoML framework for a specific domain (e.g., computer vision, natural language processing).

### Exercise 7: Interpretable AutoML
Implement interpretability metrics and constraints in the AutoML pipeline.

### Exercise 8: Production AutoML System
Build a production-ready AutoML system with model versioning, monitoring, and automated retraining.

### Exercise 9: Distributed AutoML
Implement distributed hyperparameter optimization using multiple machines.

### Exercise 10: AutoML Benchmarking
Create a benchmarking framework to compare different AutoML approaches on various datasets.

### Exercise 11: Custom Model Integration
Implement custom machine learning models and integrate them into the AutoML framework.

### Exercise 12: AutoML for Edge Computing
Optimize the AutoML framework for resource-constrained environments like mobile devices.

### Exercise 13: Automated Feature Engineering
Extend the framework to include automated feature engineering capabilities.

### Exercise 14: Meta-Learning for AutoML
Implement meta-learning techniques to improve AutoML performance based on dataset characteristics.

### Exercise 15: Real-World AutoML Application
Apply the AutoML framework to a real-world problem and document the results and insights.

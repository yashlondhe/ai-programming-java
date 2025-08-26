package com.aiprogramming.ch03;

import java.util.Map;
import java.util.List;

/**
 * Interface for machine learning algorithms.
 */
public interface MLAlgorithm {
    
    /**
     * Trains the model on the given dataset
     */
    TrainedModel train(Dataset dataset);
    
    /**
     * Trains the model with hyperparameters
     */
    default TrainedModel train(Dataset dataset, Map<String, Object> hyperparameters) {
        setHyperparameters(hyperparameters);
        return train(dataset);
    }
    
    /**
     * Sets hyperparameters for the algorithm
     */
    default void setHyperparameters(Map<String, Object> hyperparameters) {
        // Default implementation does nothing
    }
}

/**
 * Simple classifier for demonstration purposes
 */
class SimpleClassifier implements MLAlgorithm {
    
    @Override
    public TrainedModel train(Dataset dataset) {
        // Simple implementation that returns a basic trained model
        return new SimpleTrainedModel();
    }
}

/**
 * Logistic regression implementation
 */
class LogisticRegression implements MLAlgorithm {
    
    @Override
    public TrainedModel train(Dataset dataset) {
        // Placeholder implementation
        return new SimpleTrainedModel();
    }
}

/**
 * Random forest implementation
 */
class RandomForest implements MLAlgorithm {
    
    @Override
    public TrainedModel train(Dataset dataset) {
        // Placeholder implementation
        return new SimpleTrainedModel();
    }
}

/**
 * Support vector machine implementation
 */
class SupportVectorMachine implements MLAlgorithm {
    
    @Override
    public TrainedModel train(Dataset dataset) {
        // Placeholder implementation
        return new SimpleTrainedModel();
    }
}

/**
 * Polynomial regression implementation
 */
class PolynomialRegression implements MLAlgorithm {
    
    private int degree;
    
    public PolynomialRegression(int degree) {
        this.degree = degree;
    }
    
    @Override
    public TrainedModel train(Dataset dataset) {
        return new PolynomialTrainedModel(degree);
    }
    
    public double evaluate(Dataset dataset) {
        // Simple evaluation - return random error for demonstration
        return Math.random() * 0.1 + 0.05;
    }
}

/**
 * Trained model interface
 */
interface TrainedModel {
    
    /**
     * Makes a prediction on a single data point
     */
    Prediction predict(DataPoint dataPoint);
    
    /**
     * Makes predictions on a dataset
     */
    default List<Prediction> predict(Dataset dataset) {
        return dataset.getDataPoints().stream()
                .map(this::predict)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * Evaluates the model on a dataset
     */
    default double evaluate(Dataset dataset) {
        List<Prediction> predictions = predict(dataset);
        // Simple evaluation - return accuracy for classification
        long correct = predictions.stream()
                .filter(p -> p.getPredictedLabel().equals(p.getActualLabel()))
                .count();
        return (double) correct / predictions.size();
    }
    
    /**
     * Checks if the model is ready for prediction
     */
    default boolean isReady() {
        return true;
    }
}

/**
 * Simple trained model implementation
 */
class SimpleTrainedModel implements TrainedModel {
    
    @Override
    public Prediction predict(DataPoint dataPoint) {
        // Simple prediction - return random result for demonstration
        boolean predicted = Math.random() > 0.5;
        boolean actual = false;
        
        if (dataPoint.hasTarget()) {
            Object target = dataPoint.getTarget();
            if (target instanceof Boolean) {
                actual = (Boolean) target;
            } else if (target instanceof Number) {
                actual = ((Number) target).intValue() != 0;
            } else if (target instanceof String) {
                actual = "true".equalsIgnoreCase((String) target) || "1".equals(target);
            }
        }
        
        return new Prediction(predicted, actual, Math.random());
    }
}

/**
 * Polynomial trained model implementation
 */
class PolynomialTrainedModel implements TrainedModel {
    
    private final int degree;
    
    public PolynomialTrainedModel(int degree) {
        this.degree = degree;
    }
    
    @Override
    public Prediction predict(DataPoint dataPoint) {
        // Simple polynomial prediction
        double x = dataPoint.getNumericalFeature("x");
        double predicted = Math.pow(x, degree) * 0.1 + Math.random() * 0.1;
        double actual = 0.0;
        
        if (dataPoint.hasTarget()) {
            Object target = dataPoint.getTarget();
            if (target instanceof Number) {
                actual = ((Number) target).doubleValue();
            }
        }
        
        return new Prediction(predicted, actual, 0.8);
    }
}

/**
 * Prediction class
 */
class Prediction {
    private final Object predictedLabel;
    private final Object actualLabel;
    private final double probability;
    
    public Prediction(Object predictedLabel, Object actualLabel, double probability) {
        this.predictedLabel = predictedLabel;
        this.actualLabel = actualLabel;
        this.probability = probability;
    }
    
    public Object getPredictedLabel() {
        return predictedLabel;
    }
    
    public Object getActualLabel() {
        return actualLabel;
    }
    
    public double getProbability() {
        return probability;
    }
    
    public double getValue() {
        if (predictedLabel instanceof Number) {
            return ((Number) predictedLabel).doubleValue();
        }
        return 0.0;
    }
}

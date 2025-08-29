package com.aiprogramming.ch18;

import java.util.*;

/**
 * Configuration for model selection
 */
public class ModelSelectionConfig {
    
    public enum EvaluationMetric {
        ACCURACY,
        PRECISION,
        RECALL,
        F1_SCORE,
        ROC_AUC,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        R2_SCORE
    }
    
    private final Set<Class<? extends MLModel>> candidateModels;
    private final EvaluationMetric evaluationMetric;
    private final boolean useCrossValidation;
    private final int cvFolds;
    
    public ModelSelectionConfig() {
        this(new HashSet<>(Arrays.asList(
            LinearRegression.class,
            LogisticRegression.class,
            RandomForest.class,
            NeuralNetwork.class
        )), EvaluationMetric.ACCURACY, true, 5);
    }
    
    public ModelSelectionConfig(Set<Class<? extends MLModel>> candidateModels, 
                              EvaluationMetric evaluationMetric, 
                              boolean useCrossValidation, int cvFolds) {
        this.candidateModels = candidateModels;
        this.evaluationMetric = evaluationMetric;
        this.useCrossValidation = useCrossValidation;
        this.cvFolds = cvFolds;
    }
    
    public Set<Class<? extends MLModel>> getCandidateModels() {
        return candidateModels;
    }
    
    public EvaluationMetric getEvaluationMetric() {
        return evaluationMetric;
    }
    
    public boolean isUseCrossValidation() {
        return useCrossValidation;
    }
    
    public int getCvFolds() {
        return cvFolds;
    }
}

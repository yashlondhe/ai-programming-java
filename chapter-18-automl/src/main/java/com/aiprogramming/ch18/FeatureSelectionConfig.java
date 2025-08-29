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
    
    public SelectionMethod getSelectionMethod() {
        return selectionMethod;
    }
    
    public int getMaxFeatures() {
        return maxFeatures;
    }
    
    public double getThreshold() {
        return threshold;
    }
    
    public boolean isUseCrossValidation() {
        return useCrossValidation;
    }
}

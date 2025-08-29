package com.aiprogramming.ch18;

/**
 * Result of feature selection
 */
public class FeatureSelectionResult {
    
    private final int[] selectedFeatureIndices;
    private final double[][] selectedFeatures;
    private final double[] importanceScores;
    
    public FeatureSelectionResult(int[] selectedFeatureIndices, double[][] selectedFeatures, 
                                double[] importanceScores) {
        this.selectedFeatureIndices = selectedFeatureIndices;
        this.selectedFeatures = selectedFeatures;
        this.importanceScores = importanceScores;
    }
    
    public int[] getSelectedFeatureIndices() {
        return selectedFeatureIndices;
    }
    
    public double[][] getSelectedFeatures() {
        return selectedFeatures;
    }
    
    public double[] getImportanceScores() {
        return importanceScores;
    }
    
    /**
     * Print summary of feature selection results
     */
    public void printSummary() {
        System.out.println("Selected " + selectedFeatureIndices.length + " features out of " + 
                          (selectedFeatureIndices.length > 0 ? selectedFeatures[0].length + selectedFeatureIndices.length : 0));
        System.out.println("Selected feature indices: " + java.util.Arrays.toString(selectedFeatureIndices));
        
        if (importanceScores.length > 0) {
            System.out.println("Top 5 feature importance scores:");
            for (int i = 0; i < Math.min(5, importanceScores.length); i++) {
                System.out.printf("Feature %d: %.4f%n", selectedFeatureIndices[i], importanceScores[i]);
            }
        }
    }
}

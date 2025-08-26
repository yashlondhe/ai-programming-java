package com.aiprogramming.ch03;

/**
 * Represents a split of a dataset into training, validation, and test sets.
 */
public class DatasetSplit {
    private final Dataset trainingSet;
    private final Dataset validationSet;
    private final Dataset testSet;
    
    /**
     * Creates a new dataset split
     */
    public DatasetSplit(Dataset trainingSet, Dataset validationSet, Dataset testSet) {
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
        this.testSet = testSet;
    }
    
    /**
     * Gets the training set
     */
    public Dataset getTrainingSet() {
        return trainingSet;
    }
    
    /**
     * Gets the validation set
     */
    public Dataset getValidationSet() {
        return validationSet;
    }
    
    /**
     * Gets the test set
     */
    public Dataset getTestSet() {
        return testSet;
    }
    
    /**
     * Gets the total size of all sets
     */
    public int getTotalSize() {
        return trainingSet.size() + validationSet.size() + testSet.size();
    }
    
    /**
     * Gets the training set size
     */
    public int getTrainingSize() {
        return trainingSet.size();
    }
    
    /**
     * Gets the validation set size
     */
    public int getValidationSize() {
        return validationSet.size();
    }
    
    /**
     * Gets the test set size
     */
    public int getTestSize() {
        return testSet.size();
    }
    
    @Override
    public String toString() {
        return String.format("DatasetSplit{training=%d, validation=%d, test=%d}", 
                trainingSet.size(), validationSet.size(), testSet.size());
    }
}

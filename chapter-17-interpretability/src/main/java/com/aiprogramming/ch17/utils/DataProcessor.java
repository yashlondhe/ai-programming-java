package com.aiprogramming.ch17.utils;

import com.aiprogramming.utils.DataUtils;
import com.aiprogramming.utils.MatrixUtils;
import com.aiprogramming.utils.ValidationUtils;
import java.util.List;
import java.util.Random;

/**
 * Data processor for handling data loading and preprocessing
 * 
 * This class provides utilities for loading sample data, managing features,
 * and preprocessing data for interpretability analysis.
 */
public class DataProcessor {
    
    private final Random random;
    private final String[] featureNames;
    
    /**
     * Constructor for data processor
     */
    public DataProcessor() {
        this.random = new Random(42);
        this.featureNames = generateFeatureNames();
    }
    
    /**
     * Generate sample feature names for demonstration
     * 
     * @return array of feature names
     */
    private String[] generateFeatureNames() {
        return new String[]{
            "age",
            "income",
            "education_level",
            "credit_score",
            "employment_years",
            "loan_amount",
            "debt_to_income_ratio",
            "payment_history",
            "collateral_value",
            "loan_purpose"
        };
    }
    
    /**
     * Load sample data for demonstration
     * 
     * @return 2D array of sample data
     */
    public double[][] loadSampleData() {
        int numSamples = 1000;
        int numFeatures = featureNames.length;
        
        double[][] data = MatrixUtils.zeros(numSamples, numFeatures);
        
        for (int i = 0; i < numSamples; i++) {
            data[i] = generateSampleInstance();
        }
        
        return data;
    }
    
    /**
     * Generate a single sample instance
     * 
     * @return array of feature values
     */
    private double[] generateSampleInstance() {
        double[] instance = new double[featureNames.length];
        
        // Generate realistic feature values
        instance[0] = 25 + random.nextDouble() * 50; // age: 25-75
        instance[1] = 20000 + random.nextDouble() * 180000; // income: 20k-200k
        instance[2] = random.nextDouble() * 4; // education: 0-4 (high school to PhD)
        instance[3] = 300 + random.nextDouble() * 500; // credit score: 300-800
        instance[4] = random.nextDouble() * 20; // employment years: 0-20
        instance[5] = 5000 + random.nextDouble() * 495000; // loan amount: 5k-500k
        instance[6] = random.nextDouble() * 0.8; // debt to income ratio: 0-0.8
        instance[7] = random.nextDouble(); // payment history: 0-1
        instance[8] = 10000 + random.nextDouble() * 990000; // collateral value: 10k-1M
        instance[9] = random.nextDouble() * 5; // loan purpose: 0-5 (different purposes)
        
        // Normalize all features to [0, 1] range
        normalizeInstance(instance);
        
        return instance;
    }
    
    /**
     * Normalize an instance to [0, 1] range
     * 
     * @param instance the instance to normalize
     */
    private void normalizeInstance(double[] instance) {
        ValidationUtils.validateVector(instance, "instance");
        
        // Simple min-max normalization
        double min = 0.0;
        double max = 1.0;
        
        for (int i = 0; i < instance.length; i++) {
            instance[i] = Math.max(min, Math.min(max, instance[i]));
        }
    }
    
    /**
     * Load data from CSV file
     * 
     * @param filePath path to the CSV file
     * @param hasHeader whether the CSV has a header row
     * @param featureColumns indices of feature columns
     * @param targetColumn index of target column
     * @return pair of feature matrix and target vector
     * @throws Exception if file cannot be read
     */
    public DataUtils.Pair<double[][], double[]> loadDataFromCSV(String filePath, boolean hasHeader, 
                                                               int[] featureColumns, int targetColumn) throws Exception {
        ValidationUtils.validateNotNull(filePath, "filePath");
        ValidationUtils.validateNonEmptyString(filePath, "filePath");
        ValidationUtils.validateNotNull(featureColumns, "featureColumns");
        if (targetColumn < 0) {
            throw new IllegalArgumentException("Target column must be non-negative");
        }
        
        List<String[]> rawData = DataUtils.loadCSV(filePath, hasHeader);
        return DataUtils.convertToNumeric(rawData, featureColumns, targetColumn);
    }
    
    /**
     * Normalize data using min-max scaling
     * 
     * @param data input data matrix
     * @return normalized data matrix
     */
    public double[][] normalizeData(double[][] data) {
        ValidationUtils.validateMatrix(data, "data");
        return DataUtils.normalize(data);
    }
    
    /**
     * Standardize data using z-score normalization
     * 
     * @param data input data matrix
     * @return standardized data matrix
     */
    public double[][] standardizeData(double[][] data) {
        ValidationUtils.validateMatrix(data, "data");
        return DataUtils.standardize(data);
    }
    
    /**
     * Get feature names
     * 
     * @return array of feature names
     */
    public String[] getFeatureNames() {
        return featureNames.clone();
    }
    
    /**
     * Get feature name by index
     * 
     * @param index feature index
     * @return feature name
     */
    public String getFeatureName(int index) {
        if (index >= 0 && index < featureNames.length) {
            return featureNames[index];
        }
        throw new IllegalArgumentException("Invalid feature index: " + index);
    }
    
    /**
     * Get feature index by name
     * 
     * @param name feature name
     * @return feature index
     */
    public int getFeatureIndex(String name) {
        for (int i = 0; i < featureNames.length; i++) {
            if (featureNames[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Feature not found: " + name);
    }
    
    /**
     * Check if a feature exists
     * 
     * @param name feature name
     * @return true if feature exists
     */
    public boolean hasFeature(String name) {
        for (String featureName : featureNames) {
            if (featureName.equals(name)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Get number of features
     * 
     * @return number of features
     */
    public int getNumFeatures() {
        return featureNames.length;
    }
    
    /**
     * Generate synthetic data with specific characteristics
     * 
     * @param numSamples number of samples to generate
     * @param biasStrength strength of bias to introduce (0-1)
     * @return synthetic dataset
     */
    public double[][] generateSyntheticData(int numSamples, double biasStrength) {
        double[][] data = new double[numSamples][featureNames.length];
        
        for (int i = 0; i < numSamples; i++) {
            data[i] = generateBiasedInstance(biasStrength);
        }
        
        return data;
    }
    
    /**
     * Generate a biased instance
     * 
     * @param biasStrength strength of bias
     * @return biased instance
     */
    private double[] generateBiasedInstance(double biasStrength) {
        double[] instance = generateSampleInstance();
        
        // Introduce bias based on certain features
        // For example, bias based on age and income
        if (random.nextDouble() < biasStrength) {
            // Introduce systematic bias
            instance[0] *= 0.8; // Reduce age
            instance[1] *= 0.7; // Reduce income
        }
        
        return instance;
    }
    
    /**
     * Split data into training and test sets
     * 
     * @param data full dataset
     * @param trainRatio ratio of training data (0-1)
     * @return array containing [trainingData, testData]
     */
    public double[][][] splitData(double[][] data, double trainRatio) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        
        int numSamples = data.length;
        int trainSize = (int) (numSamples * trainRatio);
        int testSize = numSamples - trainSize;
        
        // Shuffle data
        double[][] shuffledData = shuffleData(data);
        
        // Split data
        double[][] trainingData = new double[trainSize][];
        double[][] testData = new double[testSize][];
        
        System.arraycopy(shuffledData, 0, trainingData, 0, trainSize);
        System.arraycopy(shuffledData, trainSize, testData, 0, testSize);
        
        return new double[][][]{trainingData, testData};
    }
    
    /**
     * Shuffle the data
     * 
     * @param data data to shuffle
     * @return shuffled data
     */
    private double[][] shuffleData(double[][] data) {
        double[][] shuffled = new double[data.length][];
        System.arraycopy(data, 0, shuffled, 0, data.length);
        
        // Fisher-Yates shuffle
        for (int i = shuffled.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double[] temp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = temp;
        }
        
        return shuffled;
    }
    
    /**
     * Calculate basic statistics for a dataset
     * 
     * @param data dataset to analyze
     * @return statistics for each feature
     */
    public double[][] calculateStatistics(double[][] data) {
        if (data == null || data.length == 0) {
            return new double[0][0];
        }
        
        int numFeatures = data[0].length;
        double[][] stats = new double[numFeatures][4]; // min, max, mean, std
        
        for (int j = 0; j < numFeatures; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            double sum = 0.0;
            double sumSq = 0.0;
            
            for (double[] instance : data) {
                double value = instance[j];
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
                sumSq += value * value;
            }
            
            double mean = sum / data.length;
            double variance = (sumSq / data.length) - (mean * mean);
            double std = Math.sqrt(variance);
            
            stats[j][0] = min;
            stats[j][1] = max;
            stats[j][2] = mean;
            stats[j][3] = std;
        }
        
        return stats;
    }
    
    /**
     * Print dataset statistics
     * 
     * @param data dataset to analyze
     */
    public void printStatistics(double[][] data) {
        double[][] stats = calculateStatistics(data);
        
        System.out.println("Dataset Statistics:");
        System.out.printf("%-20s %-10s %-10s %-10s %-10s%n", 
                         "Feature", "Min", "Max", "Mean", "Std");
        System.out.println("-".repeat(60));
        
        for (int i = 0; i < featureNames.length; i++) {
            System.out.printf("%-20s %-10.4f %-10.4f %-10.4f %-10.4f%n",
                             featureNames[i], stats[i][0], stats[i][1], stats[i][2], stats[i][3]);
        }
    }
}

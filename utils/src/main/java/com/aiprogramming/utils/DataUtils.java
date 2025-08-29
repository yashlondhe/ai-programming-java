package com.aiprogramming.utils;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility class for data processing operations commonly used in AI/ML applications.
 * Provides methods for data loading, preprocessing, validation, and transformation.
 */
public class DataUtils {
    
    /**
     * Loads CSV data from a file and returns it as a list of string arrays.
     * 
     * @param filePath Path to the CSV file
     * @param hasHeader Whether the CSV has a header row
     * @return List of string arrays representing rows
     * @throws IOException If file cannot be read
     */
    public static List<String[]> loadCSV(String filePath, boolean hasHeader) throws IOException {
        List<String[]> data = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean firstLine = true;
            
            while ((line = reader.readLine()) != null) {
                if (hasHeader && firstLine) {
                    firstLine = false;
                    continue;
                }
                
                String[] row = line.split(",");
                data.add(row);
                firstLine = false;
            }
        }
        
        return data;
    }
    
    /**
     * Converts string data to numeric arrays for machine learning.
     * 
     * @param data List of string arrays
     * @param featureColumns Indices of feature columns
     * @param targetColumn Index of target column
     * @return Pair of feature matrix and target vector
     */
    public static Pair<double[][], double[]> convertToNumeric(List<String[]> data, 
                                                             int[] featureColumns, 
                                                             int targetColumn) {
        int rows = data.size();
        int features = featureColumns.length;
        
        double[][] X = new double[rows][features];
        double[] y = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            String[] row = data.get(i);
            
            // Extract features
            for (int j = 0; j < features; j++) {
                try {
                    X[i][j] = Double.parseDouble(row[featureColumns[j]]);
                } catch (NumberFormatException e) {
                    X[i][j] = 0.0; // Default value for non-numeric data
                }
            }
            
            // Extract target
            try {
                y[i] = Double.parseDouble(row[targetColumn]);
            } catch (NumberFormatException e) {
                y[i] = 0.0;
            }
        }
        
        return new Pair<>(X, y);
    }
    
    /**
     * Normalizes data using min-max scaling to range [0, 1].
     * 
     * @param data Input data matrix
     * @return Normalized data matrix
     */
    public static double[][] normalize(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] normalized = new double[rows][cols];
        
        for (int j = 0; j < cols; j++) {
            // Find min and max for this column
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            
            for (int i = 0; i < rows; i++) {
                min = Math.min(min, data[i][j]);
                max = Math.max(max, data[i][j]);
            }
            
            // Normalize column
            double range = max - min;
            if (range == 0) {
                range = 1; // Avoid division by zero
            }
            
            for (int i = 0; i < rows; i++) {
                normalized[i][j] = (data[i][j] - min) / range;
            }
        }
        
        return normalized;
    }
    
    /**
     * Standardizes data using z-score normalization (mean=0, std=1).
     * 
     * @param data Input data matrix
     * @return Standardized data matrix
     */
    public static double[][] standardize(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] standardized = new double[rows][cols];
        
        for (int j = 0; j < cols; j++) {
            // Calculate mean
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            double mean = sum / rows;
            
            // Calculate standard deviation
            double sumSquared = 0;
            for (int i = 0; i < rows; i++) {
                sumSquared += Math.pow(data[i][j] - mean, 2);
            }
            double std = Math.sqrt(sumSquared / rows);
            
            // Standardize column
            if (std == 0) {
                std = 1; // Avoid division by zero
            }
            
            for (int i = 0; i < rows; i++) {
                standardized[i][j] = (data[i][j] - mean) / std;
            }
        }
        
        return standardized;
    }
    
    /**
     * Splits data into training and testing sets.
     * 
     * @param X Feature matrix
     * @param y Target vector
     * @param testSize Proportion of data for testing (0.0 to 1.0)
     * @param randomSeed Random seed for reproducibility
     * @return TrainTestSplit object containing split data
     */
    public static TrainTestSplit trainTestSplit(double[][] X, double[] y, 
                                               double testSize, long randomSeed) {
        int totalSamples = X.length;
        int testSamples = (int) (totalSamples * testSize);
        int trainSamples = totalSamples - testSamples;
        
        // Create indices and shuffle
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        
        Random random = new Random(randomSeed);
        Collections.shuffle(indices, random);
        
        // Split data
        double[][] XTrain = new double[trainSamples][X[0].length];
        double[][] XTest = new double[testSamples][X[0].length];
        double[] yTrain = new double[trainSamples];
        double[] yTest = new double[testSamples];
        
        for (int i = 0; i < trainSamples; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < testSamples; i++) {
            int idx = indices.get(trainSamples + i);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }
        
        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
    
    /**
     * Calculates accuracy for classification predictions.
     * 
     * @param yTrue True labels
     * @param yPred Predicted labels
     * @return Accuracy score (0.0 to 1.0)
     */
    public static double accuracy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        
        return (double) correct / yTrue.length;
    }
    
    /**
     * Calculates mean squared error for regression predictions.
     * 
     * @param yTrue True values
     * @param yPred Predicted values
     * @return Mean squared error
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        double sum = 0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return sum / yTrue.length;
    }
    
    /**
     * Calculates root mean squared error for regression predictions.
     * 
     * @param yTrue True values
     * @param yPred Predicted values
     * @return Root mean squared error
     */
    public static double rootMeanSquaredError(double[] yTrue, double[] yPred) {
        return Math.sqrt(meanSquaredError(yTrue, yPred));
    }
    
    /**
     * Calculates R-squared (coefficient of determination) for regression.
     * 
     * @param yTrue True values
     * @param yPred Predicted values
     * @return R-squared score
     */
    public static double rSquared(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        // Calculate mean of true values
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        
        // Calculate total sum of squares and residual sum of squares
        double tss = 0; // Total sum of squares
        double rss = 0; // Residual sum of squares
        
        for (int i = 0; i < yTrue.length; i++) {
            tss += Math.pow(yTrue[i] - mean, 2);
            rss += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1 - (rss / tss);
    }
    
    /**
     * Prints a matrix to console for debugging.
     * 
     * @param matrix Matrix to print
     * @param name Name of the matrix
     */
    public static void printMatrix(double[][] matrix, String name) {
        System.out.println(name + ":");
        for (int i = 0; i < Math.min(matrix.length, 10); i++) { // Limit to first 10 rows
            for (int j = 0; j < Math.min(matrix[i].length, 10); j++) { // Limit to first 10 cols
                System.out.printf("%.4f ", matrix[i][j]);
            }
            if (matrix[i].length > 10) {
                System.out.print("...");
            }
            System.out.println();
        }
        if (matrix.length > 10) {
            System.out.println("...");
        }
        System.out.println("Shape: " + matrix.length + " x " + matrix[0].length);
    }
    
    /**
     * Prints a vector to console for debugging.
     * 
     * @param vector Vector to print
     * @param name Name of the vector
     */
    public static void printVector(double[] vector, String name) {
        System.out.println(name + ":");
        for (int i = 0; i < Math.min(vector.length, 20); i++) {
            System.out.printf("%.4f ", vector[i]);
        }
        if (vector.length > 20) {
            System.out.print("...");
        }
        System.out.println();
        System.out.println("Length: " + vector.length);
    }
    
    /**
     * Simple Pair class for returning multiple values.
     */
    public static class Pair<T, U> {
        public final T first;
        public final U second;
        
        public Pair(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }
    
    /**
     * Container class for train-test split results.
     */
    public static class TrainTestSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;
        
        public TrainTestSplit(double[][] XTrain, double[][] XTest, 
                            double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
}

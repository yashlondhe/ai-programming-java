package com.aiprogramming.ch02;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Java Fundamentals Demo for AI Development
 * 
 * This class demonstrates essential Java concepts that are particularly
 * important for AI and machine learning development.
 */
public class JavaFundamentalsDemo {
    
    private static final Logger logger = LoggerFactory.getLogger(JavaFundamentalsDemo.class);
    
    public static void main(String[] args) {
        logger.info("Java Fundamentals for AI Development");
        logger.info("=======================================\n");
        
        // Demonstrate collections for AI data
        demonstrateCollectionsForAI();
        
        // Show Stream API and functional programming
        demonstrateStreamAPI();
        
        // Explore matrix operations
        demonstrateMatrixOperations();
        
        // Performance optimization techniques
        demonstratePerformanceOptimization();
        
        // Exception handling for AI applications
        demonstrateExceptionHandling();
        
        logger.info("Java fundamentals demonstration completed!");
    }
    
    /**
     * Demonstrates Java collections framework for AI data handling
     */
    private static void demonstrateCollectionsForAI() {
        logger.info("Collections Framework for AI Data:");
        
        // 1. Lists for sequential data (e.g., time series, feature vectors)
        logger.info("\n1. Lists for Sequential Data:");
        List<Double> featureVector = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0);
        logger.info("   Feature vector: {}", featureVector);
        
        // Calculate mean using traditional approach
        double sum = 0.0;
        for (Double value : featureVector) {
            sum += value;
        }
        double mean = sum / featureVector.size();
        logger.info("   Mean (traditional): {:.2f}", mean);
        
        // 2. Sets for unique elements (e.g., unique classes, features)
        logger.info("\n2. Sets for Unique Elements:");
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList("cat", "dog", "cat", "bird", "dog"));
        logger.info("   Unique classes: {}", uniqueClasses);
        
        // 3. Maps for key-value pairs (e.g., feature importance, word frequencies)
        logger.info("\n3. Maps for Key-Value Pairs:");
        Map<String, Double> featureImportance = new HashMap<>();
        featureImportance.put("age", 0.8);
        featureImportance.put("income", 0.6);
        featureImportance.put("education", 0.4);
        logger.info("   Feature importance: {}", featureImportance);
        
        // 4. Queues for processing order (e.g., batch processing)
        logger.info("\n4. Queues for Processing Order:");
        Queue<String> processingQueue = new LinkedList<>();
        processingQueue.offer("data_point_1");
        processingQueue.offer("data_point_2");
        processingQueue.offer("data_point_3");
        logger.info("   Processing queue: {}", processingQueue);
        logger.info("   Next to process: {}", processingQueue.poll());
        
        // 5. Multidimensional arrays for matrices
        logger.info("\n5. Multidimensional Arrays for Matrices:");
        double[][] matrix = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        logger.info("   3x3 Matrix:");
        for (double[] row : matrix) {
            logger.info("   {}", Arrays.toString(row));
        }
    }
    
    /**
     * Demonstrates Stream API and functional programming concepts
     */
    private static void demonstrateStreamAPI() {
        logger.info("\n🔄 Stream API and Functional Programming:");
        
        // Sample dataset: student scores
        List<Student> students = Arrays.asList(
            new Student("Alice", 85.5),
            new Student("Bob", 92.0),
            new Student("Charlie", 78.5),
            new Student("Diana", 95.0),
            new Student("Eve", 88.5)
        );
        
        // 1. Filtering data
        logger.info("\n1. Filtering Data:");
        List<Student> highPerformers = students.stream()
            .filter(student -> student.getScore() >= 90.0)
            .collect(Collectors.toList());
        logger.info("   High performers (≥90): {}", highPerformers);
        
        // 2. Mapping transformations
        logger.info("\n2. Mapping Transformations:");
        List<String> names = students.stream()
            .map(Student::getName)
            .collect(Collectors.toList());
        logger.info("   Student names: {}", names);
        
        // 3. Aggregation operations
        logger.info("\n3. Aggregation Operations:");
        double averageScore = students.stream()
            .mapToDouble(Student::getScore)
            .average()
            .orElse(0.0);
        logger.info("   Average score: {:.2f}", averageScore);
        
        double maxScore = students.stream()
            .mapToDouble(Student::getScore)
            .max()
            .orElse(0.0);
        logger.info("   Maximum score: {:.2f}", maxScore);
        
        // 4. Complex data processing pipeline
        logger.info("\n4. Complex Data Processing Pipeline:");
        Map<String, Double> scoreMap = students.stream()
            .filter(student -> student.getScore() >= 80.0)
            .collect(Collectors.toMap(
                Student::getName,
                Student::getScore
            ));
        logger.info("   Students with scores ≥80: {}", scoreMap);
        
        // 5. Parallel processing
        logger.info("\n5. Parallel Processing:");
        long startTime = System.currentTimeMillis();
        double parallelSum = students.parallelStream()
            .mapToDouble(Student::getScore)
            .sum();
        long endTime = System.currentTimeMillis();
        logger.info("   Parallel sum: {:.2f} (took {}ms)", parallelSum, endTime - startTime);
    }
    
    /**
     * Demonstrates matrix operations essential for AI algorithms
     */
    private static void demonstrateMatrixOperations() {
        logger.info("\n🧮 Matrix Operations for AI:");
        
        // 1. Matrix creation and initialization
        logger.info("\n1. Matrix Creation:");
        int rows = 3, cols = 3;
        double[][] matrixA = new double[rows][cols];
        double[][] matrixB = new double[rows][cols];
        
        // Initialize matrices
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixA[i][j] = i + j;
                matrixB[i][j] = i * j;
            }
        }
        
        logger.info("   Matrix A:");
        printMatrix(matrixA);
        logger.info("   Matrix B:");
        printMatrix(matrixB);
        
        // 2. Matrix addition
        logger.info("\n2. Matrix Addition:");
        double[][] sumMatrix = addMatrices(matrixA, matrixB);
        logger.info("   A + B:");
        printMatrix(sumMatrix);
        
        // 3. Matrix multiplication
        logger.info("\n3. Matrix Multiplication:");
        double[][] productMatrix = multiplyMatrices(matrixA, matrixB);
        logger.info("   A × B:");
        printMatrix(productMatrix);
        
        // 4. Transpose operation
        logger.info("\n4. Matrix Transpose:");
        double[][] transposeMatrix = transposeMatrix(matrixA);
        logger.info("   A^T:");
        printMatrix(transposeMatrix);
        
        // 5. Vector operations
        logger.info("\n5. Vector Operations:");
        double[] vector1 = {1.0, 2.0, 3.0};
        double[] vector2 = {4.0, 5.0, 6.0};
        
        double dotProduct = dotProduct(vector1, vector2);
        logger.info("   Dot product: {:.2f}", dotProduct);
        
        double[] crossProduct = crossProduct(vector1, vector2);
        logger.info("   Cross product: {}", Arrays.toString(crossProduct));
    }
    
    /**
     * Demonstrates performance optimization techniques
     */
    private static void demonstratePerformanceOptimization() {
        logger.info("\n⚡ Performance Optimization:");
        
        // 1. Memory management
        logger.info("\n1. Memory Management:");
        int largeSize = 1000000;
        
        // Inefficient approach
        long startTime = System.currentTimeMillis();
        List<Integer> inefficientList = new ArrayList<>();
        for (int i = 0; i < largeSize; i++) {
            inefficientList.add(i);
        }
        long endTime = System.currentTimeMillis();
        logger.info("   Inefficient list creation: {}ms", endTime - startTime);
        
        // Efficient approach with initial capacity
        startTime = System.currentTimeMillis();
        List<Integer> efficientList = new ArrayList<>(largeSize);
        for (int i = 0; i < largeSize; i++) {
            efficientList.add(i);
        }
        endTime = System.currentTimeMillis();
        logger.info("   Efficient list creation: {}ms", endTime - startTime);
        
        // 2. String concatenation optimization
        logger.info("\n2. String Concatenation:");
        int iterations = 10000;
        
        // Inefficient string concatenation
        startTime = System.currentTimeMillis();
        String inefficientString = "";
        for (int i = 0; i < iterations; i++) {
            inefficientString += "item" + i + " ";
        }
        endTime = System.currentTimeMillis();
        logger.info("   Inefficient concatenation: {}ms", endTime - startTime);
        
        // Efficient string concatenation with StringBuilder
        startTime = System.currentTimeMillis();
        StringBuilder efficientString = new StringBuilder();
        for (int i = 0; i < iterations; i++) {
            efficientString.append("item").append(i).append(" ");
        }
        endTime = System.currentTimeMillis();
        logger.info("   Efficient concatenation: {}ms", endTime - startTime);
        
        // 3. Boxing/unboxing optimization
        logger.info("\n3. Boxing/Unboxing Optimization:");
        
        // Inefficient boxing
        startTime = System.currentTimeMillis();
        List<Integer> boxedList = new ArrayList<>();
        for (int i = 0; i < 100000; i++) {
            boxedList.add(i); // Autoboxing
        }
        endTime = System.currentTimeMillis();
        logger.info("   Boxing overhead: {}ms", endTime - startTime);
        
        // Efficient primitive arrays
        startTime = System.currentTimeMillis();
        int[] primitiveArray = new int[100000];
        for (int i = 0; i < 100000; i++) {
            primitiveArray[i] = i; // No boxing
        }
        endTime = System.currentTimeMillis();
        logger.info("   Primitive array: {}ms", endTime - startTime);
    }
    
    /**
     * Demonstrates exception handling for AI applications
     */
    private static void demonstrateExceptionHandling() {
        logger.info("\n🛡️ Exception Handling for AI:");
        
        // 1. Custom exceptions for AI applications
        logger.info("\n1. Custom AI Exceptions:");
        try {
            validateFeatureVector(new double[]{1.0, 2.0, -1.0, 4.0});
        } catch (InvalidFeatureVectorException e) {
            logger.error("   Feature vector validation failed: {}", e.getMessage());
        }
        
        // 2. Resource management with try-with-resources
        logger.info("\n2. Resource Management:");
        try (DataProcessor processor = new DataProcessor()) {
            processor.processData("sample_data.csv");
        } catch (DataProcessingException e) {
            logger.error("   Data processing failed: {}", e.getMessage());
        }
        
        // 3. Graceful degradation
        logger.info("\n3. Graceful Degradation:");
        double result = safeDivision(10.0, 0.0);
        logger.info("   Safe division result: {}", result);
        
        // 4. Batch processing with error recovery
        logger.info("\n4. Batch Processing with Error Recovery:");
        List<String> dataPoints = Arrays.asList("valid_data", "invalid_data", "valid_data");
        List<String> processedData = new ArrayList<>();
        
        for (String dataPoint : dataPoints) {
            try {
                String processed = processDataPoint(dataPoint);
                processedData.add(processed);
            } catch (DataProcessingException e) {
                logger.warn("   Skipping invalid data point: {}", dataPoint);
                // Continue processing other data points
            }
        }
        logger.info("   Successfully processed: {} out of {}", processedData.size(), dataPoints.size());
    }
    
    // Helper methods for matrix operations
    
    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            logger.info("   {}", Arrays.toString(row));
        }
    }
    
    private static double[][] addMatrices(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    
    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }
    
    private static double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    private static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double result = 0.0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    private static double[] crossProduct(double[] a, double[] b) {
        if (a.length != 3 || b.length != 3) {
            throw new IllegalArgumentException("Cross product only defined for 3D vectors");
        }
        
        return new double[]{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
    
    // Helper methods for exception handling
    
    private static void validateFeatureVector(double[] features) throws InvalidFeatureVectorException {
        for (int i = 0; i < features.length; i++) {
            if (features[i] < 0) {
                throw new InvalidFeatureVectorException("Negative value at index " + i + ": " + features[i]);
            }
        }
    }
    
    private static double safeDivision(double numerator, double denominator) {
        try {
            return numerator / denominator;
        } catch (ArithmeticException e) {
            logger.warn("   Division by zero detected, returning 0.0");
            return 0.0;
        }
    }
    
    private static String processDataPoint(String dataPoint) throws DataProcessingException {
        if ("invalid_data".equals(dataPoint)) {
            throw new DataProcessingException("Invalid data point: " + dataPoint);
        }
        return "processed_" + dataPoint;
    }
    
    // Inner classes and custom exceptions
    
    static class Student {
        private final String name;
        private final double score;
        
        public Student(String name, double score) {
            this.name = name;
            this.score = score;
        }
        
        public String getName() { return name; }
        public double getScore() { return score; }
        
        @Override
        public String toString() {
            return name + "(" + score + ")";
        }
    }
    
    static class InvalidFeatureVectorException extends Exception {
        public InvalidFeatureVectorException(String message) {
            super(message);
        }
    }
    
    static class DataProcessingException extends Exception {
        public DataProcessingException(String message) {
            super(message);
        }
    }
    
    static class DataProcessor implements AutoCloseable {
        public void processData(String filename) throws DataProcessingException {
            // Simulate data processing
            if (filename.contains("invalid")) {
                throw new DataProcessingException("Cannot process invalid file: " + filename);
            }
            logger.info("   Processing data from: {}", filename);
        }
        
        @Override
        public void close() {
            logger.info("   Closing data processor");
        }
    }
}

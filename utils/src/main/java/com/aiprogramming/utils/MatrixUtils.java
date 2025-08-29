package com.aiprogramming.utils;

import java.util.Arrays;
import java.util.Random;

/**
 * Utility class for matrix operations commonly used in AI/ML applications.
 * Provides methods for matrix creation, manipulation, and mathematical operations.
 */
public class MatrixUtils {
    
    /**
     * Creates a matrix filled with zeros.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Zero matrix
     */
    public static double[][] zeros(int rows, int cols) {
        return new double[rows][cols];
    }
    
    /**
     * Creates a matrix filled with ones.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Matrix filled with ones
     */
    public static double[][] ones(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            Arrays.fill(matrix[i], 1.0);
        }
        return matrix;
    }
    
    /**
     * Creates an identity matrix.
     * 
     * @param size Size of the square matrix
     * @return Identity matrix
     */
    public static double[][] identity(int size) {
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            matrix[i][i] = 1.0;
        }
        return matrix;
    }
    
    /**
     * Creates a matrix with random values from uniform distribution.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param min Minimum value
     * @param max Maximum value
     * @param seed Random seed for reproducibility
     * @return Random matrix
     */
    public static double[][] random(int rows, int cols, double min, double max, long seed) {
        Random random = new Random(seed);
        double[][] matrix = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = min + (max - min) * random.nextDouble();
            }
        }
        
        return matrix;
    }
    
    /**
     * Creates a matrix with random values from normal distribution.
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param mean Mean of the normal distribution
     * @param std Standard deviation of the normal distribution
     * @param seed Random seed for reproducibility
     * @return Random matrix
     */
    public static double[][] randomNormal(int rows, int cols, double mean, double std, long seed) {
        Random random = new Random(seed);
        double[][] matrix = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = mean + std * random.nextGaussian();
            }
        }
        
        return matrix;
    }
    
    /**
     * Transposes a matrix.
     * 
     * @param matrix Input matrix
     * @return Transposed matrix
     */
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        
        return transposed;
    }
    
    /**
     * Multiplies two matrices.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Result of matrix multiplication
     */
    public static double[][] multiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int rowsB = B.length;
        int colsB = B[0].length;
        
        if (colsA != rowsB) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Multiplies a matrix by a scalar.
     * 
     * @param matrix Input matrix
     * @param scalar Scalar value
     * @return Result matrix
     */
    public static double[][] multiply(double[][] matrix, double scalar) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        
        return result;
    }
    
    /**
     * Adds two matrices element-wise.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Result matrix
     */
    public static double[][] add(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        
        if (rows != B.length || cols != B[0].length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for addition");
        }
        
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Subtracts matrix B from matrix A element-wise.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Result matrix
     */
    public static double[][] subtract(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        
        if (rows != B.length || cols != B[0].length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for subtraction");
        }
        
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Performs element-wise multiplication (Hadamard product).
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Result matrix
     */
    public static double[][] elementWiseMultiply(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        
        if (rows != B.length || cols != B[0].length) {
            throw new IllegalArgumentException("Matrix dimensions don't match for element-wise multiplication");
        }
        
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] * B[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Calculates the sum of all elements in a matrix.
     * 
     * @param matrix Input matrix
     * @return Sum of all elements
     */
    public static double sum(double[][] matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j];
            }
        }
        return sum;
    }
    
    /**
     * Calculates the mean of all elements in a matrix.
     * 
     * @param matrix Input matrix
     * @return Mean of all elements
     */
    public static double mean(double[][] matrix) {
        int totalElements = matrix.length * matrix[0].length;
        return sum(matrix) / totalElements;
    }
    
    /**
     * Calculates the maximum value in a matrix.
     * 
     * @param matrix Input matrix
     * @return Maximum value
     */
    public static double max(double[][] matrix) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                max = Math.max(max, matrix[i][j]);
            }
        }
        return max;
    }
    
    /**
     * Calculates the minimum value in a matrix.
     * 
     * @param matrix Input matrix
     * @return Minimum value
     */
    public static double min(double[][] matrix) {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                min = Math.min(min, matrix[i][j]);
            }
        }
        return min;
    }
    
    /**
     * Applies a function to each element of a matrix.
     * 
     * @param matrix Input matrix
     * @param function Function to apply
     * @return Result matrix
     */
    public static double[][] apply(double[][] matrix, MatrixFunction function) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = function.apply(matrix[i][j]);
            }
        }
        
        return result;
    }
    
    /**
     * Calculates the dot product of two vectors.
     * 
     * @param a First vector
     * @param b Second vector
     * @return Dot product
     */
    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector lengths don't match");
        }
        
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        
        return result;
    }
    
    /**
     * Calculates the Euclidean norm (magnitude) of a vector.
     * 
     * @param vector Input vector
     * @return Euclidean norm
     */
    public static double norm(double[] vector) {
        double sum = 0;
        for (double value : vector) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Normalizes a vector to unit length.
     * 
     * @param vector Input vector
     * @return Normalized vector
     */
    public static double[] normalize(double[] vector) {
        double norm = norm(vector);
        if (norm == 0) {
            return vector.clone();
        }
        
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        return normalized;
    }
    
    /**
     * Creates a copy of a matrix.
     * 
     * @param matrix Input matrix
     * @return Copy of the matrix
     */
    public static double[][] copy(double[][] matrix) {
        int rows = matrix.length;
        double[][] copy = new double[rows][];
        
        for (int i = 0; i < rows; i++) {
            copy[i] = matrix[i].clone();
        }
        
        return copy;
    }
    
    /**
     * Checks if two matrices are equal within a tolerance.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @param tolerance Tolerance for comparison
     * @return True if matrices are equal within tolerance
     */
    public static boolean equals(double[][] A, double[][] B, double tolerance) {
        if (A.length != B.length || A[0].length != B[0].length) {
            return false;
        }
        
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                if (Math.abs(A[i][j] - B[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Functional interface for matrix element operations.
     */
    @FunctionalInterface
    public interface MatrixFunction {
        double apply(double value);
    }
    
    /**
     * Common matrix functions.
     */
    public static class Functions {
        public static final MatrixFunction SIGMOID = x -> 1.0 / (1.0 + Math.exp(-x));
        public static final MatrixFunction TANH = x -> Math.tanh(x);
        public static final MatrixFunction RELU = x -> Math.max(0, x);
        public static final MatrixFunction EXP = x -> Math.exp(x);
        public static final MatrixFunction LOG = x -> Math.log(x);
        public static final MatrixFunction SQUARE = x -> x * x;
        public static final MatrixFunction SQRT = x -> Math.sqrt(x);
    }
}

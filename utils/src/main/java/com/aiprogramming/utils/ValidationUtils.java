package com.aiprogramming.utils;

import java.util.Arrays;

/**
 * Utility class for validation and error checking in AI/ML applications.
 * Provides methods for input validation, data quality checks, and error handling.
 */
public class ValidationUtils {
    
    /**
     * Validates that a matrix is not null and has valid dimensions.
     * 
     * @param matrix Matrix to validate
     * @param name Name of the matrix for error messages
     * @throws IllegalArgumentException If matrix is invalid
     */
    public static void validateMatrix(double[][] matrix, String name) {
        if (matrix == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        
        if (matrix.length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
        
        if (matrix[0] == null) {
            throw new IllegalArgumentException(name + " contains null rows");
        }
        
        int expectedCols = matrix[0].length;
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i] == null) {
                throw new IllegalArgumentException(name + " contains null rows");
            }
            if (matrix[i].length != expectedCols) {
                throw new IllegalArgumentException(name + " has inconsistent column counts");
            }
        }
    }
    
    /**
     * Validates that a vector is not null and has valid length.
     * 
     * @param vector Vector to validate
     * @param name Name of the vector for error messages
     * @throws IllegalArgumentException If vector is invalid
     */
    public static void validateVector(double[] vector, String name) {
        if (vector == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        
        if (vector.length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
    }
    
    /**
     * Validates that two matrices have compatible dimensions for multiplication.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @param operation Description of the operation for error messages
     * @throws IllegalArgumentException If matrices are incompatible
     */
    public static void validateMatrixMultiplication(double[][] A, double[][] B, String operation) {
        validateMatrix(A, "Matrix A");
        validateMatrix(B, "Matrix B");
        
        if (A[0].length != B.length) {
            throw new IllegalArgumentException(
                "Matrix dimensions incompatible for " + operation + 
                ": A(" + A.length + "x" + A[0].length + ") * B(" + B.length + "x" + B[0].length + ")");
        }
    }
    
    /**
     * Validates that two matrices have the same dimensions.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @param operation Description of the operation for error messages
     * @throws IllegalArgumentException If matrices have different dimensions
     */
    public static void validateSameDimensions(double[][] A, double[][] B, String operation) {
        validateMatrix(A, "Matrix A");
        validateMatrix(B, "Matrix B");
        
        if (A.length != B.length || A[0].length != B[0].length) {
            throw new IllegalArgumentException(
                "Matrix dimensions must match for " + operation + 
                ": A(" + A.length + "x" + A[0].length + ") vs B(" + B.length + "x" + B[0].length + ")");
        }
    }
    
    /**
     * Validates that a vector has the expected length.
     * 
     * @param vector Vector to validate
     * @param expectedLength Expected length
     * @param name Name of the vector for error messages
     * @throws IllegalArgumentException If vector has wrong length
     */
    public static void validateVectorLength(double[] vector, int expectedLength, String name) {
        validateVector(vector, name);
        
        if (vector.length != expectedLength) {
            throw new IllegalArgumentException(
                name + " must have length " + expectedLength + ", but has length " + vector.length);
        }
    }
    
    /**
     * Validates that a matrix has the expected dimensions.
     * 
     * @param matrix Matrix to validate
     * @param expectedRows Expected number of rows
     * @param expectedCols Expected number of columns
     * @param name Name of the matrix for error messages
     * @throws IllegalArgumentException If matrix has wrong dimensions
     */
    public static void validateMatrixDimensions(double[][] matrix, int expectedRows, int expectedCols, String name) {
        validateMatrix(matrix, name);
        
        if (matrix.length != expectedRows || matrix[0].length != expectedCols) {
            throw new IllegalArgumentException(
                name + " must have dimensions " + expectedRows + "x" + expectedCols + 
                ", but has dimensions " + matrix.length + "x" + matrix[0].length);
        }
    }
    
    /**
     * Validates that a value is within a specified range.
     * 
     * @param value Value to validate
     * @param min Minimum allowed value
     * @param max Maximum allowed value
     * @param name Name of the value for error messages
     * @throws IllegalArgumentException If value is out of range
     */
    public static void validateRange(double value, double min, double max, String name) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(
                name + " must be between " + min + " and " + max + ", but is " + value);
        }
    }
    
    /**
     * Validates that a value is positive.
     * 
     * @param value Value to validate
     * @param name Name of the value for error messages
     * @throws IllegalArgumentException If value is not positive
     */
    public static void validatePositive(double value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive, but is " + value);
        }
    }
    
    /**
     * Validates that a value is non-negative.
     * 
     * @param value Value to validate
     * @param name Name of the value for error messages
     * @throws IllegalArgumentException If value is negative
     */
    public static void validateNonNegative(double value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, but is " + value);
        }
    }
    
    /**
     * Validates that a probability value is between 0 and 1.
     * 
     * @param value Probability value to validate
     * @param name Name of the value for error messages
     * @throws IllegalArgumentException If value is not a valid probability
     */
    public static void validateProbability(double value, String name) {
        validateRange(value, 0.0, 1.0, name);
    }
    
    /**
     * Validates that an array contains no null elements.
     * 
     * @param array Array to validate
     * @param name Name of the array for error messages
     * @throws IllegalArgumentException If array contains null elements
     */
    public static void validateNoNulls(Object[] array, String name) {
        if (array == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
        
        for (int i = 0; i < array.length; i++) {
            if (array[i] == null) {
                throw new IllegalArgumentException(name + " contains null element at index " + i);
            }
        }
    }
    
    /**
     * Validates that a matrix contains no NaN or infinite values.
     * 
     * @param matrix Matrix to validate
     * @param name Name of the matrix for error messages
     * @throws IllegalArgumentException If matrix contains invalid values
     */
    public static void validateFiniteValues(double[][] matrix, String name) {
        validateMatrix(matrix, name);
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (Double.isNaN(matrix[i][j]) || Double.isInfinite(matrix[i][j])) {
                    throw new IllegalArgumentException(
                        name + " contains invalid value at position [" + i + "][" + j + "]: " + matrix[i][j]);
                }
            }
        }
    }
    
    /**
     * Validates that a vector contains no NaN or infinite values.
     * 
     * @param vector Vector to validate
     * @param name Name of the vector for error messages
     * @throws IllegalArgumentException If vector contains invalid values
     */
    public static void validateFiniteValues(double[] vector, String name) {
        validateVector(vector, name);
        
        for (int i = 0; i < vector.length; i++) {
            if (Double.isNaN(vector[i]) || Double.isInfinite(vector[i])) {
                throw new IllegalArgumentException(
                    name + " contains invalid value at index " + i + ": " + vector[i]);
            }
        }
    }
    
    /**
     * Validates that a matrix is square.
     * 
     * @param matrix Matrix to validate
     * @param name Name of the matrix for error messages
     * @throws IllegalArgumentException If matrix is not square
     */
    public static void validateSquare(double[][] matrix, String name) {
        validateMatrix(matrix, name);
        
        if (matrix.length != matrix[0].length) {
            throw new IllegalArgumentException(
                name + " must be square, but has dimensions " + matrix.length + "x" + matrix[0].length);
        }
    }
    
    /**
     * Validates that a matrix is symmetric.
     * 
     * @param matrix Matrix to validate
     * @param name Name of the matrix for error messages
     * @param tolerance Tolerance for floating-point comparison
     * @throws IllegalArgumentException If matrix is not symmetric
     */
    public static void validateSymmetric(double[][] matrix, String name, double tolerance) {
        validateSquare(matrix, name);
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i + 1; j < matrix.length; j++) {
                if (Math.abs(matrix[i][j] - matrix[j][i]) > tolerance) {
                    throw new IllegalArgumentException(
                        name + " is not symmetric: matrix[" + i + "][" + j + "] = " + matrix[i][j] + 
                        " != matrix[" + j + "][" + i + "] = " + matrix[j][i]);
                }
            }
        }
    }
    
    /**
     * Validates that a matrix is positive definite (for covariance matrices).
     * This is a simplified check - for production use, consider using a library.
     * 
     * @param matrix Matrix to validate
     * @param name Name of the matrix for error messages
     * @throws IllegalArgumentException If matrix is not positive definite
     */
    public static void validatePositiveDefinite(double[][] matrix, String name) {
        validateSquare(matrix, name);
        validateSymmetric(matrix, name, 1e-10);
        
        // Check that diagonal elements are positive
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][i] <= 0) {
                throw new IllegalArgumentException(
                    name + " is not positive definite: diagonal element at [" + i + "][" + i + "] = " + matrix[i][i]);
            }
        }
    }
    
    /**
     * Validates that arrays have the same length.
     * 
     * @param arrays Arrays to validate
     * @param names Names of the arrays for error messages
     * @throws IllegalArgumentException If arrays have different lengths
     */
    public static void validateSameLength(double[][] arrays, String[] names) {
        if (arrays.length != names.length) {
            throw new IllegalArgumentException("Number of arrays and names must match");
        }
        
        if (arrays.length < 2) {
            return; // Nothing to validate
        }
        
        int expectedLength = arrays[0].length;
        for (int i = 1; i < arrays.length; i++) {
            if (arrays[i].length != expectedLength) {
                throw new IllegalArgumentException(
                    "All arrays must have the same length: " + names[0] + " has length " + expectedLength + 
                    ", but " + names[i] + " has length " + arrays[i].length);
            }
        }
    }
    
    /**
     * Validates that a string is not null or empty.
     * 
     * @param str String to validate
     * @param name Name of the string for error messages
     * @throws IllegalArgumentException If string is null or empty
     */
    public static void validateNonEmptyString(String str, String name) {
        if (str == null || str.trim().isEmpty()) {
            throw new IllegalArgumentException(name + " cannot be null or empty");
        }
    }
    
    /**
     * Validates that an object is not null.
     * 
     * @param obj Object to validate
     * @param name Name of the object for error messages
     * @throws IllegalArgumentException If object is null
     */
    public static void validateNotNull(Object obj, String name) {
        if (obj == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
    }
    
    /**
     * Validates that an array is not empty.
     * 
     * @param array Array to validate
     * @param name Name of the array for error messages
     * @throws IllegalArgumentException If array is empty
     */
    public static void validateNonEmpty(Object[] array, String name) {
        validateNotNull(array, name);
        
        if (array.length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
    }
    
    /**
     * Validates that a collection is not empty.
     * 
     * @param collection Collection to validate
     * @param name Name of the collection for error messages
     * @throws IllegalArgumentException If collection is empty
     */
    public static void validateNonEmpty(java.util.Collection<?> collection, String name) {
        validateNotNull(collection, name);
        
        if (collection.isEmpty()) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
    }
}

package com.aiprogramming.ch08;

import java.util.Arrays;
import java.util.Random;

/**
 * Multi-dimensional tensor for CNN operations
 */
public class Tensor {
    private final int[] shape;
    private final double[] data;
    private final int totalSize;
    
    /**
     * Create a tensor with the specified shape
     */
    public Tensor(int... shape) {
        this.shape = shape.clone();
        this.totalSize = calculateTotalSize(shape);
        this.data = new double[totalSize];
    }
    
    /**
     * Create a tensor with the specified shape and data
     */
    public Tensor(double[] data, int... shape) {
        this.shape = shape.clone();
        this.totalSize = calculateTotalSize(shape);
        if (data.length != totalSize) {
            throw new IllegalArgumentException("Data length doesn't match shape");
        }
        this.data = data.clone();
    }
    
    /**
     * Create a tensor filled with random values
     */
    public static Tensor random(int... shape) {
        Tensor tensor = new Tensor(shape);
        Random random = new Random();
        for (int i = 0; i < tensor.totalSize; i++) {
            tensor.data[i] = random.nextGaussian() * 0.1;
        }
        return tensor;
    }
    
    /**
     * Create a tensor filled with zeros
     */
    public static Tensor zeros(int... shape) {
        Tensor tensor = new Tensor(shape);
        Arrays.fill(tensor.data, 0.0);
        return tensor;
    }
    
    /**
     * Create a tensor filled with ones
     */
    public static Tensor ones(int... shape) {
        Tensor tensor = new Tensor(shape);
        Arrays.fill(tensor.data, 1.0);
        return tensor;
    }
    
    /**
     * Get value at specific indices
     */
    public double get(int... indices) {
        int index = calculateIndex(indices);
        return data[index];
    }
    
    /**
     * Set value at specific indices
     */
    public void set(double value, int... indices) {
        int index = calculateIndex(indices);
        data[index] = value;
    }
    
    /**
     * Get the shape of the tensor
     */
    public int[] getShape() {
        return shape.clone();
    }
    
    /**
     * Get the total number of elements
     */
    public int getTotalSize() {
        return totalSize;
    }
    
    /**
     * Get the underlying data array
     */
    public double[] getData() {
        return data.clone();
    }
    
    /**
     * Add another tensor to this tensor
     */
    public Tensor add(Tensor other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shapes must match for addition");
        }
        
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < totalSize; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }
    
    /**
     * Multiply tensor by a scalar
     */
    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < totalSize; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }
    
    /**
     * Element-wise multiplication with another tensor
     */
    public Tensor multiply(Tensor other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shapes must match for multiplication");
        }
        
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < totalSize; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }
    
    /**
     * Reshape the tensor
     */
    public Tensor reshape(int... newShape) {
        int newTotalSize = calculateTotalSize(newShape);
        if (newTotalSize != totalSize) {
            throw new IllegalArgumentException("New shape must have same total size");
        }
        
        Tensor result = new Tensor(this.data, newShape);
        return result;
    }
    
    /**
     * Get a slice of the tensor
     */
    public Tensor slice(int startIndex, int endIndex, int dimension) {
        if (dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds");
        }
        
        int[] newShape = shape.clone();
        newShape[dimension] = endIndex - startIndex;
        
        Tensor result = new Tensor(newShape);
        // Implementation would copy the appropriate slice
        // This is a simplified version
        return result;
    }
    
    /**
     * Calculate the index in the flat array for given indices
     */
    private int calculateIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Number of indices must match tensor dimensions");
        }
        
        int index = 0;
        int multiplier = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IllegalArgumentException("Index out of bounds");
            }
            index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        
        return index;
    }
    
    /**
     * Calculate total size from shape
     */
    private int calculateTotalSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Shape dimensions must be positive");
            }
            size *= dim;
        }
        return size;
    }
    
    /**
     * Convert to string representation
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape)).append(")");
        if (totalSize <= 20) {
            sb.append(" data=").append(Arrays.toString(data));
        } else {
            sb.append(" data=[...]");
        }
        return sb.toString();
    }
    
    /**
     * Check equality with another tensor
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Tensor tensor = (Tensor) obj;
        return Arrays.equals(shape, tensor.shape) && Arrays.equals(data, tensor.data);
    }
    
    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }
}

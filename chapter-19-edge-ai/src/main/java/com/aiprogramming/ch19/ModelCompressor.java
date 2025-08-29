package com.aiprogramming.ch19;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Model compression utilities for edge AI deployment
 * Implements pruning, quantization, and knowledge distillation techniques
 */
public class ModelCompressor {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelCompressor.class);
    
    /**
     * Pruning configuration
     */
    public static class PruningConfig {
        private final double sparsity; // Target sparsity (0.0 = no pruning, 1.0 = all weights zero)
        private final String method; // "magnitude", "structured", "dynamic"
        private final boolean retrain; // Whether to retrain after pruning
        
        public PruningConfig(double sparsity, String method, boolean retrain) {
            this.sparsity = sparsity;
            this.method = method;
            this.retrain = retrain;
        }
        
        public double getSparsity() { return sparsity; }
        public String getMethod() { return method; }
        public boolean isRetrain() { return retrain; }
    }
    
    /**
     * Quantization configuration
     */
    public static class QuantizationConfig {
        private final int bits; // Number of bits (8, 16, 32)
        private final String method; // "uniform", "non-uniform", "dynamic"
        private final boolean symmetric; // Whether to use symmetric quantization
        
        public QuantizationConfig(int bits, String method, boolean symmetric) {
            this.bits = bits;
            this.method = method;
            this.symmetric = symmetric;
        }
        
        public int getBits() { return bits; }
        public String getMethod() { return method; }
        public boolean isSymmetric() { return symmetric; }
    }
    
    /**
     * Compressed model representation
     */
    public static class CompressedModel {
        private final RealMatrix weights;
        private final double[] biases;
        private final Map<String, Object> metadata;
        private final double compressionRatio;
        private final double accuracyLoss;
        
        public CompressedModel(RealMatrix weights, double[] biases, 
                             Map<String, Object> metadata, 
                             double compressionRatio, double accuracyLoss) {
            this.weights = weights;
            this.biases = biases;
            this.metadata = metadata;
            this.compressionRatio = compressionRatio;
            this.accuracyLoss = accuracyLoss;
        }
        
        public RealMatrix getWeights() { return weights; }
        public double[] getBiases() { return biases; }
        public Map<String, Object> getMetadata() { return metadata; }
        public double getCompressionRatio() { return compressionRatio; }
        public double getAccuracyLoss() { return accuracyLoss; }
    }
    
    /**
     * Magnitude-based pruning
     * Removes weights with smallest absolute values
     */
    public static RealMatrix magnitudePruning(RealMatrix weights, double sparsity) {
        logger.info("Applying magnitude pruning with sparsity: {}", sparsity);
        
        int rows = weights.getRowDimension();
        int cols = weights.getColumnDimension();
        int totalWeights = rows * cols;
        int weightsToKeep = (int) Math.round(totalWeights * (1 - sparsity));
        
        // Flatten weights and get indices of largest absolute values
        double[] flatWeights = new double[totalWeights];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flatWeights[idx++] = Math.abs(weights.getEntry(i, j));
            }
        }
        
        // Find threshold for top weights
        Arrays.sort(flatWeights);
        double threshold = flatWeights[totalWeights - weightsToKeep];
        
        // Apply pruning
        RealMatrix prunedWeights = weights.copy();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(weights.getEntry(i, j)) < threshold) {
                    prunedWeights.setEntry(i, j, 0.0);
                }
            }
        }
        
        double actualSparsity = 1.0 - (double) countNonZeros(prunedWeights) / totalWeights;
        logger.info("Actual sparsity achieved: {}", String.format("%.3f", actualSparsity));
        
        return prunedWeights;
    }
    
    /**
     * Structured pruning - removes entire rows/columns
     */
    public static RealMatrix structuredPruning(RealMatrix weights, double sparsity) {
        logger.info("Applying structured pruning with sparsity: {}", sparsity);
        
        int rows = weights.getRowDimension();
        int cols = weights.getColumnDimension();
        
        // Calculate row and column importance scores
        double[] rowScores = new double[rows];
        double[] colScores = new double[cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double weight = weights.getEntry(i, j);
                rowScores[i] += weight * weight;
                colScores[j] += weight * weight;
            }
        }
        
        // Find rows and columns to prune
        int rowsToPrune = (int) Math.round(rows * sparsity / 2);
        int colsToPrune = (int) Math.round(cols * sparsity / 2);
        
        // Get indices of least important rows and columns
        int[] rowIndices = getTopIndices(rowScores, rowsToPrune, false);
        int[] colIndices = getTopIndices(colScores, colsToPrune, false);
        
        // Create pruned matrix
        RealMatrix prunedWeights = new Array2DRowRealMatrix(rows - rowsToPrune, cols - colsToPrune);
        int newRow = 0;
        for (int i = 0; i < rows; i++) {
            if (!contains(rowIndices, i)) {
                int newCol = 0;
                for (int j = 0; j < cols; j++) {
                    if (!contains(colIndices, j)) {
                        prunedWeights.setEntry(newRow, newCol, weights.getEntry(i, j));
                        newCol++;
                    }
                }
                newRow++;
            }
        }
        
        return prunedWeights;
    }
    
    /**
     * Uniform quantization
     * Maps float values to integer representation
     */
    public static QuantizedMatrix uniformQuantization(RealMatrix weights, int bits) {
        logger.info("Applying uniform quantization with {} bits", bits);
        
        int rows = weights.getRowDimension();
        int cols = weights.getColumnDimension();
        
        // Find min and max values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double weight = weights.getEntry(i, j);
                min = Math.min(min, weight);
                max = Math.max(max, weight);
            }
        }
        
        // Calculate scale and zero point
        double scale = (max - min) / (Math.pow(2, bits) - 1);
        double zeroPoint = min;
        
        // Quantize weights
        int[][] quantizedWeights = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double weight = weights.getEntry(i, j);
                int quantized = (int) Math.round((weight - zeroPoint) / scale);
                quantized = Math.max(0, Math.min(quantized, (int) Math.pow(2, bits) - 1));
                quantizedWeights[i][j] = quantized;
            }
        }
        
        return new QuantizedMatrix(quantizedWeights, scale, zeroPoint);
    }
    
    /**
     * Knowledge distillation
     * Uses a larger teacher model to train a smaller student model
     */
    public static RealMatrix knowledgeDistillation(RealMatrix teacherWeights, 
                                                 RealMatrix studentWeights, 
                                                 double temperature, 
                                                 double alpha) {
        logger.info("Applying knowledge distillation with temperature: {}, alpha: {}", temperature, alpha);
        
        // Simple implementation: soft target transfer
        RealMatrix distilledWeights = studentWeights.copy();
        
        // Apply soft targets from teacher (only for overlapping dimensions)
        int minRows = Math.min(teacherWeights.getRowDimension(), studentWeights.getRowDimension());
        int minCols = Math.min(teacherWeights.getColumnDimension(), studentWeights.getColumnDimension());
        
        for (int i = 0; i < minRows; i++) {
            for (int j = 0; j < minCols; j++) {
                double teacherWeight = teacherWeights.getEntry(i, j);
                double studentWeight = studentWeights.getEntry(i, j);
                
                // Soft target: weighted combination
                double softTarget = alpha * teacherWeight + (1 - alpha) * studentWeight;
                distilledWeights.setEntry(i, j, softTarget);
            }
        }
        
        return distilledWeights;
    }
    
    /**
     * Comprehensive model compression pipeline
     */
    public static CompressedModel compressModel(RealMatrix originalWeights, 
                                              double[] originalBiases,
                                              PruningConfig pruningConfig,
                                              QuantizationConfig quantizationConfig) {
        logger.info("Starting model compression pipeline");
        
        RealMatrix compressedWeights = originalWeights.copy();
        double[] compressedBiases = originalBiases.clone();
        
        // Step 1: Pruning
        if (pruningConfig.getSparsity() > 0) {
            if ("magnitude".equals(pruningConfig.getMethod())) {
                compressedWeights = magnitudePruning(compressedWeights, pruningConfig.getSparsity());
            } else if ("structured".equals(pruningConfig.getMethod())) {
                compressedWeights = structuredPruning(compressedWeights, pruningConfig.getSparsity());
            }
        }
        
        // Step 2: Quantization
        QuantizedMatrix quantizedMatrix = null;
        if (quantizationConfig.getBits() < 32) {
            quantizedMatrix = uniformQuantization(compressedWeights, quantizationConfig.getBits());
            // Convert back to RealMatrix for compatibility
            compressedWeights = quantizedMatrix.toRealMatrix();
        }
        
        // Calculate compression metrics
        double originalSize = originalWeights.getRowDimension() * originalWeights.getColumnDimension() * 4; // 4 bytes per float
        double compressedSize = countNonZeros(compressedWeights) * 4;
        if (quantizedMatrix != null) {
            compressedSize = compressedWeights.getRowDimension() * compressedWeights.getColumnDimension() * (quantizationConfig.getBits() / 8.0);
        }
        
        double compressionRatio = originalSize / compressedSize;
        
        // Metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("pruning_config", pruningConfig);
        metadata.put("quantization_config", quantizationConfig);
        metadata.put("original_shape", new int[]{originalWeights.getRowDimension(), originalWeights.getColumnDimension()});
        metadata.put("compressed_shape", new int[]{compressedWeights.getRowDimension(), compressedWeights.getColumnDimension()});
        metadata.put("sparsity", 1.0 - (double) countNonZeros(compressedWeights) / (compressedWeights.getRowDimension() * compressedWeights.getColumnDimension()));
        
        logger.info("Compression completed. Ratio: {}x", String.format("%.2f", compressionRatio));
        
        return new CompressedModel(compressedWeights, compressedBiases, metadata, compressionRatio, 0.0);
    }
    
    // Helper methods
    private static int countNonZeros(RealMatrix matrix) {
        int count = 0;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (Math.abs(matrix.getEntry(i, j)) > 1e-10) {
                    count++;
                }
            }
        }
        return count;
    }
    
    private static int[] getTopIndices(double[] values, int count, boolean top) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            indices.add(i);
        }
        
        if (top) {
            indices.sort((a, b) -> Double.compare(values[b], values[a]));
        } else {
            indices.sort((a, b) -> Double.compare(values[a], values[b]));
        }
        
        return indices.subList(0, count).stream().mapToInt(Integer::intValue).toArray();
    }
    
    private static boolean contains(int[] array, int value) {
        for (int item : array) {
            if (item == value) return true;
        }
        return false;
    }
    
    /**
     * Quantized matrix representation
     */
    public static class QuantizedMatrix {
        private final int[][] weights;
        private final double scale;
        private final double zeroPoint;
        
        public QuantizedMatrix(int[][] weights, double scale, double zeroPoint) {
            this.weights = weights;
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }
        
        public RealMatrix toRealMatrix() {
            int rows = weights.length;
            int cols = weights[0].length;
            RealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double dequantized = weights[i][j] * scale + zeroPoint;
                    matrix.setEntry(i, j, dequantized);
                }
            }
            
            return matrix;
        }
        
        public int[][] getWeights() { return weights; }
        public double getScale() { return scale; }
        public double getZeroPoint() { return zeroPoint; }
    }
}

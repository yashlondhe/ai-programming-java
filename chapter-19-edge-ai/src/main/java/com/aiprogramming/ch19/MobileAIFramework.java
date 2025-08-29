package com.aiprogramming.ch19;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.io.*;

/**
 * Mobile AI Framework for edge device deployment
 * Provides optimization and deployment utilities for mobile AI applications
 */
public class MobileAIFramework {
    
    private static final Logger logger = LoggerFactory.getLogger(MobileAIFramework.class);
    
    /**
     * Mobile device constraints
     */
    public static class DeviceConstraints {
        private final int maxMemoryMB;
        private final int maxModelSizeMB;
        private final double maxInferenceTimeMs;
        private final int maxBatteryDrain;
        private final boolean supportsGPU;
        
        public DeviceConstraints(int maxMemoryMB, int maxModelSizeMB, double maxInferenceTimeMs, 
                               int maxBatteryDrain, boolean supportsGPU) {
            this.maxMemoryMB = maxMemoryMB;
            this.maxModelSizeMB = maxModelSizeMB;
            this.maxInferenceTimeMs = maxInferenceTimeMs;
            this.maxBatteryDrain = maxBatteryDrain;
            this.supportsGPU = supportsGPU;
        }
        
        public int getMaxMemoryMB() { return maxMemoryMB; }
        public int getMaxModelSizeMB() { return maxModelSizeMB; }
        public double getMaxInferenceTimeMs() { return maxInferenceTimeMs; }
        public int getMaxBatteryDrain() { return maxBatteryDrain; }
        public boolean supportsGPU() { return supportsGPU; }
    }
    
    /**
     * Mobile-optimized model
     */
    public static class MobileModel {
        private final RealMatrix weights;
        private final double[] biases;
        private final Map<String, Object> metadata;
        private final DeviceConstraints constraints;
        private final double modelSizeMB;
        private final double inferenceTimeMs;
        
        public MobileModel(RealMatrix weights, double[] biases, Map<String, Object> metadata,
                          DeviceConstraints constraints, double modelSizeMB, double inferenceTimeMs) {
            this.weights = weights;
            this.biases = biases;
            this.metadata = metadata;
            this.constraints = constraints;
            this.modelSizeMB = modelSizeMB;
            this.inferenceTimeMs = inferenceTimeMs;
        }
        
        /**
         * Make prediction with mobile-optimized inference
         */
        public double predict(double[] features) {
            long startTime = System.nanoTime();
            
            double result = 0.0;
            
            // Optimized matrix multiplication for mobile
            for (int i = 0; i < weights.getRowDimension(); i++) {
                for (int j = 0; j < weights.getColumnDimension(); j++) {
                    result += weights.getEntry(i, j) * features[j % features.length];
                }
                result += biases[i];
            }
            
            long endTime = System.nanoTime();
            double inferenceTime = (endTime - startTime) / 1_000_000.0; // Convert to milliseconds
            
            if (inferenceTime > constraints.getMaxInferenceTimeMs()) {
                logger.warn("Inference time {}ms exceeds constraint {}ms", inferenceTime, constraints.getMaxInferenceTimeMs());
            }
            
            return result;
        }
        
        /**
         * Batch prediction for efficiency
         */
        public double[] predictBatch(List<double[]> featuresList) {
            double[] predictions = new double[featuresList.size()];
            
            for (int i = 0; i < featuresList.size(); i++) {
                predictions[i] = predict(featuresList.get(i));
            }
            
            return predictions;
        }
        
        /**
         * Check if model meets device constraints
         */
        public boolean meetsConstraints() {
            boolean memoryOK = modelSizeMB <= constraints.getMaxModelSizeMB();
            boolean timeOK = inferenceTimeMs <= constraints.getMaxInferenceTimeMs();
            
            if (!memoryOK) {
                logger.warn("Model size {}MB exceeds constraint {}MB", modelSizeMB, constraints.getMaxModelSizeMB());
            }
            if (!timeOK) {
                logger.warn("Inference time {}ms exceeds constraint {}ms", inferenceTimeMs, constraints.getMaxInferenceTimeMs());
            }
            
            return memoryOK && timeOK;
        }
        
        public RealMatrix getWeights() { return weights; }
        public double[] getBiases() { return biases; }
        public Map<String, Object> getMetadata() { return metadata; }
        public double getModelSizeMB() { return modelSizeMB; }
        public double getInferenceTimeMs() { return inferenceTimeMs; }
    }
    
    /**
     * Model optimizer for mobile deployment
     */
    public static class MobileModelOptimizer {
        
        /**
         * Optimize model for mobile deployment
         */
        public static MobileModel optimizeForMobile(RealMatrix originalWeights, double[] originalBiases,
                                                  DeviceConstraints constraints) {
            logger.info("Optimizing model for mobile deployment");
            
            RealMatrix optimizedWeights = originalWeights.copy();
            double[] optimizedBiases = originalBiases.clone();
            Map<String, Object> metadata = new HashMap<>();
            
            // Step 1: Pruning for size reduction
            double targetSparsity = calculateOptimalSparsity(originalWeights, constraints);
            if (targetSparsity > 0) {
                optimizedWeights = ModelCompressor.magnitudePruning(optimizedWeights, targetSparsity);
                metadata.put("pruning_sparsity", targetSparsity);
            }
            
            // Step 2: Quantization for memory efficiency
            int optimalBits = calculateOptimalBits(optimizedWeights, constraints);
            if (optimalBits < 32) {
                ModelCompressor.QuantizedMatrix quantized = ModelCompressor.uniformQuantization(optimizedWeights, optimalBits);
                optimizedWeights = quantized.toRealMatrix();
                metadata.put("quantization_bits", optimalBits);
            }
            
            // Step 3: Calculate model metrics
            double modelSizeMB = calculateModelSize(optimizedWeights, optimizedBiases, metadata);
            double inferenceTimeMs = estimateInferenceTime(optimizedWeights, optimizedBiases);
            
            metadata.put("optimization_steps", Arrays.asList("pruning", "quantization"));
            metadata.put("original_size_mb", calculateModelSize(originalWeights, originalBiases, new HashMap<>()));
            
            return new MobileModel(optimizedWeights, optimizedBiases, metadata, constraints, modelSizeMB, inferenceTimeMs);
        }
        
        /**
         * Calculate optimal sparsity based on device constraints
         */
        private static double calculateOptimalSparsity(RealMatrix weights, DeviceConstraints constraints) {
            double originalSizeMB = calculateModelSize(weights, new double[1], new HashMap<>());
            double targetSizeMB = constraints.getMaxModelSizeMB();
            
            if (originalSizeMB <= targetSizeMB) {
                return 0.0; // No pruning needed
            }
            
            // Estimate sparsity needed to meet size constraint
            double sizeRatio = targetSizeMB / originalSizeMB;
            double estimatedSparsity = 1.0 - sizeRatio;
            
            // Cap sparsity at 0.9 to maintain model quality
            return Math.min(estimatedSparsity, 0.9);
        }
        
        /**
         * Calculate optimal quantization bits
         */
        private static int calculateOptimalBits(RealMatrix weights, DeviceConstraints constraints) {
            double originalSizeMB = calculateModelSize(weights, new double[1], new HashMap<>());
            double targetSizeMB = constraints.getMaxModelSizeMB();
            
            if (originalSizeMB <= targetSizeMB) {
                return 32; // No quantization needed
            }
            
            // Try different bit widths
            for (int bits : new int[]{8, 16}) {
                double quantizedSizeMB = originalSizeMB * bits / 32.0;
                if (quantizedSizeMB <= targetSizeMB) {
                    return bits;
                }
            }
            
            return 8; // Use 8-bit as fallback
        }
        
        /**
         * Calculate model size in MB
         */
        private static double calculateModelSize(RealMatrix weights, double[] biases, Map<String, Object> metadata) {
            int weightParams = weights.getRowDimension() * weights.getColumnDimension();
            int biasParams = biases.length;
            int totalParams = weightParams + biasParams;
            
            // Estimate bits per parameter based on metadata
            int bitsPerParam = 32; // Default to 32-bit
            if (metadata.containsKey("quantization_bits")) {
                bitsPerParam = (Integer) metadata.get("quantization_bits");
            }
            
            // Account for sparsity
            if (metadata.containsKey("pruning_sparsity")) {
                double sparsity = (Double) metadata.get("pruning_sparsity");
                totalParams = (int) (totalParams * (1 - sparsity));
            }
            
            double sizeBytes = totalParams * bitsPerParam / 8.0;
            return sizeBytes / (1024 * 1024); // Convert to MB
        }
        
        /**
         * Estimate inference time in milliseconds
         */
        private static double estimateInferenceTime(RealMatrix weights, double[] biases) {
            int operations = weights.getRowDimension() * weights.getColumnDimension() + biases.length;
            
            // Rough estimate: 1 operation = 1 nanosecond
            double timeNs = operations * 1.0;
            return timeNs / 1_000_000.0; // Convert to milliseconds
        }
    }
    
    /**
     * Mobile AI deployment manager
     */
    public static class MobileDeploymentManager {
        
        /**
         * Deploy model to mobile device
         */
        public static MobileModel deployModel(RealMatrix weights, double[] biases, DeviceConstraints constraints) {
            logger.info("Deploying model to mobile device");
            logger.info("Device constraints: {}MB memory, {}MB model size, {}ms inference time", 
                       constraints.getMaxMemoryMB(), constraints.getMaxModelSizeMB(), constraints.getMaxInferenceTimeMs());
            
            // Optimize model for mobile
            MobileModel mobileModel = MobileModelOptimizer.optimizeForMobile(weights, biases, constraints);
            
            // Validate constraints
            if (!mobileModel.meetsConstraints()) {
                logger.error("Model does not meet device constraints");
                throw new RuntimeException("Model optimization failed to meet device constraints");
            }
            
            logger.info("Model deployed successfully");
            logger.info("Final model size: {}MB, Estimated inference time: {}ms", 
                       String.format("%.2f", mobileModel.getModelSizeMB()), String.format("%.2f", mobileModel.getInferenceTimeMs()));
            
            return mobileModel;
        }
        
        /**
         * Simulate mobile inference with resource monitoring
         */
        public static class MobileInferenceSimulator {
            private final MobileModel model;
            private final DeviceConstraints constraints;
            private final List<Double> inferenceTimes;
            private final List<Double> memoryUsage;
            
            public MobileInferenceSimulator(MobileModel model, DeviceConstraints constraints) {
                this.model = model;
                this.constraints = constraints;
                this.inferenceTimes = new ArrayList<>();
                this.memoryUsage = new ArrayList<>();
            }
            
            /**
             * Simulate inference with resource monitoring
             */
            public double simulateInference(double[] features) {
                // Monitor memory usage
                Runtime runtime = Runtime.getRuntime();
                long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
                
                // Perform inference
                long startTime = System.nanoTime();
                double prediction = model.predict(features);
                long endTime = System.nanoTime();
                
                // Calculate metrics
                double inferenceTimeMs = (endTime - startTime) / 1_000_000.0;
                long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
                double memoryUsedMB = (memoryAfter - memoryBefore) / (1024.0 * 1024.0);
                
                // Store metrics
                inferenceTimes.add(inferenceTimeMs);
                memoryUsage.add(memoryUsedMB);
                
                return prediction;
            }
            
            /**
             * Get performance statistics
             */
            public Map<String, Double> getPerformanceStats() {
                Map<String, Double> stats = new HashMap<>();
                
                if (!inferenceTimes.isEmpty()) {
                    stats.put("avg_inference_time_ms", inferenceTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0));
                    stats.put("max_inference_time_ms", inferenceTimes.stream().mapToDouble(Double::doubleValue).max().orElse(0));
                    stats.put("min_inference_time_ms", inferenceTimes.stream().mapToDouble(Double::doubleValue).min().orElse(0));
                }
                
                if (!memoryUsage.isEmpty()) {
                    stats.put("avg_memory_usage_mb", memoryUsage.stream().mapToDouble(Double::doubleValue).average().orElse(0));
                    stats.put("max_memory_usage_mb", memoryUsage.stream().mapToDouble(Double::doubleValue).max().orElse(0));
                }
                
                return stats;
            }
            
            /**
             * Check if performance meets constraints
             */
            public boolean meetsPerformanceConstraints() {
                Map<String, Double> stats = getPerformanceStats();
                
                boolean timeOK = stats.getOrDefault("avg_inference_time_ms", 0.0) <= constraints.getMaxInferenceTimeMs();
                boolean memoryOK = stats.getOrDefault("max_memory_usage_mb", 0.0) <= constraints.getMaxMemoryMB();
                
                return timeOK && memoryOK;
            }
        }
    }
    
    /**
     * Predefined device profiles
     */
    public static class DeviceProfiles {
        public static final DeviceConstraints HIGH_END_PHONE = new DeviceConstraints(
            4096, // 4GB RAM
            100,  // 100MB model size
            50.0, // 50ms inference time
            10,   // 10% battery drain
            true  // GPU support
        );
        
        public static final DeviceConstraints MID_RANGE_PHONE = new DeviceConstraints(
            2048, // 2GB RAM
            50,   // 50MB model size
            100.0, // 100ms inference time
            15,   // 15% battery drain
            false // No GPU support
        );
        
        public static final DeviceConstraints LOW_END_PHONE = new DeviceConstraints(
            1024, // 1GB RAM
            25,   // 25MB model size
            200.0, // 200ms inference time
            20,   // 20% battery drain
            false // No GPU support
        );
        
        public static final DeviceConstraints IOT_DEVICE = new DeviceConstraints(
            256,  // 256MB RAM
            10,   // 10MB model size
            500.0, // 500ms inference time
            25,   // 25% battery drain
            false // No GPU support
        );
    }
}

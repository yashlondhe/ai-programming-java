package com.aiprogramming.ch19;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import com.aiprogramming.utils.MatrixUtils;
import com.aiprogramming.utils.ValidationUtils;

/**
 * Comprehensive example demonstrating Edge AI and Mobile Deployment concepts
 * Shows model compression, federated learning, and mobile optimization
 */
public class EdgeAIExample {
    
    private static final Logger logger = LoggerFactory.getLogger(EdgeAIExample.class);
    
    public static void main(String[] args) {
        logger.info("=== Edge AI and Mobile Deployment Example ===");
        
        // Generate sample model
        RealMatrix originalWeights = generateSampleModel();
        double[] originalBiases = new double[]{0.1, -0.2, 0.3};
        
        logger.info("Original model size: {}x{}", originalWeights.getRowDimension(), originalWeights.getColumnDimension());
        
        // Part 1: Model Compression
        demonstrateModelCompression(originalWeights, originalBiases);
        
        // Part 2: Federated Learning
        demonstrateFederatedLearning();
        
        // Part 3: Mobile Deployment
        demonstrateMobileDeployment(originalWeights, originalBiases);
        
        logger.info("=== Edge AI Example Completed ===");
    }
    
    /**
     * Demonstrate model compression techniques
     */
    private static void demonstrateModelCompression(RealMatrix weights, double[] biases) {
        logger.info("\n=== Model Compression Demonstration ===");
        
        // 1. Magnitude Pruning
        logger.info("1. Magnitude Pruning");
        RealMatrix prunedWeights = ModelCompressor.magnitudePruning(weights, 0.5);
        logger.info("   Pruned model: {}x{}", prunedWeights.getRowDimension(), prunedWeights.getColumnDimension());
        
        // 2. Structured Pruning
        logger.info("2. Structured Pruning");
        RealMatrix structuredPruned = ModelCompressor.structuredPruning(weights, 0.3);
        logger.info("   Structured pruned model: {}x{}", structuredPruned.getRowDimension(), structuredPruned.getColumnDimension());
        
        // 3. Quantization
        logger.info("3. Quantization");
        ModelCompressor.QuantizedMatrix quantized = ModelCompressor.uniformQuantization(weights, 8);
        logger.info("   Quantized to 8-bit precision");
        logger.info("   Scale: {}, Zero point: {}", String.format("%.6f", quantized.getScale()), String.format("%.6f", quantized.getZeroPoint()));
        
        // 4. Comprehensive Compression Pipeline
        logger.info("4. Comprehensive Compression Pipeline");
        ModelCompressor.PruningConfig pruningConfig = new ModelCompressor.PruningConfig(0.4, "magnitude", false);
        ModelCompressor.QuantizationConfig quantizationConfig = new ModelCompressor.QuantizationConfig(16, "uniform", true);
        
        ModelCompressor.CompressedModel compressed = ModelCompressor.compressModel(
            weights, biases, pruningConfig, quantizationConfig);
        
        logger.info("   Compression ratio: {}x", String.format("%.2f", compressed.getCompressionRatio()));
        logger.info("   Final model size: {}x{}", compressed.getWeights().getRowDimension(), compressed.getWeights().getColumnDimension());
        
        // 5. Knowledge Distillation
        logger.info("5. Knowledge Distillation");
        RealMatrix teacherWeights = weights.copy();
        RealMatrix studentWeights = new Array2DRowRealMatrix(weights.getRowDimension() / 2, weights.getColumnDimension() / 2);
        
        RealMatrix distilled = ModelCompressor.knowledgeDistillation(teacherWeights, studentWeights, 2.0, 0.7);
        logger.info("   Distilled model: {}x{}", distilled.getRowDimension(), distilled.getColumnDimension());
    }
    
    /**
     * Demonstrate federated learning
     */
    private static void demonstrateFederatedLearning() {
        logger.info("\n=== Federated Learning Demonstration ===");
        
        // Configuration
        int numClients = 5;
        int samplesPerClient = 100;
        int numFeatures = 10;
        int totalRounds = 3;
        int localEpochs = 5;
        double learningRate = 0.01;
        
        logger.info("Federated Learning Configuration:");
        logger.info("   Clients: {}", numClients);
        logger.info("   Samples per client: {}", samplesPerClient);
        logger.info("   Features: {}", numFeatures);
        logger.info("   Total rounds: {}", totalRounds);
        logger.info("   Local epochs: {}", localEpochs);
        logger.info("   Learning rate: {}", learningRate);
        
        // Run federated learning experiment
        FederatedLearning.FederatedServer server = FederatedLearning.runFederatedLearningExperiment(
            numClients, samplesPerClient, numFeatures, totalRounds, localEpochs, learningRate);
        
        // Test the trained global model
        logger.info("Testing global model...");
        double[] testFeatures = new double[numFeatures];
        RandomDataGenerator random = new RandomDataGenerator();
        for (int i = 0; i < numFeatures; i++) {
            testFeatures[i] = random.nextGaussian(0, 1);
        }
        
        double prediction = server.predict(testFeatures);
        logger.info("   Test prediction: {:.4f}", prediction);
        
        // Show final model parameters
        RealMatrix finalWeights = server.getGlobalWeights();
        double[] finalBiases = server.getGlobalBiases();
        logger.info("   Final model: {}x{}", finalWeights.getRowDimension(), finalWeights.getColumnDimension());
    }
    
    /**
     * Demonstrate mobile deployment
     */
    private static void demonstrateMobileDeployment(RealMatrix weights, double[] biases) {
        logger.info("\n=== Mobile Deployment Demonstration ===");
        
        // Test different device profiles
        MobileAIFramework.DeviceConstraints[] profiles = {
            MobileAIFramework.DeviceProfiles.HIGH_END_PHONE,
            MobileAIFramework.DeviceProfiles.MID_RANGE_PHONE,
            MobileAIFramework.DeviceProfiles.LOW_END_PHONE,
            MobileAIFramework.DeviceProfiles.IOT_DEVICE
        };
        
        String[] profileNames = {"High-End Phone", "Mid-Range Phone", "Low-End Phone", "IoT Device"};
        
        for (int i = 0; i < profiles.length; i++) {
            logger.info("Deploying to {}:", profileNames[i]);
            
            try {
                // Deploy model to device
                MobileAIFramework.MobileModel mobileModel = MobileAIFramework.MobileDeploymentManager.deployModel(
                    weights, biases, profiles[i]);
                
                // Create inference simulator
                MobileAIFramework.MobileDeploymentManager.MobileInferenceSimulator simulator = 
                    new MobileAIFramework.MobileDeploymentManager.MobileInferenceSimulator(mobileModel, profiles[i]);
                
                // Simulate multiple inferences
                RandomDataGenerator random = new RandomDataGenerator();
                for (int j = 0; j < 10; j++) {
                    double[] features = new double[weights.getColumnDimension()];
                    for (int k = 0; k < features.length; k++) {
                        features[k] = random.nextGaussian(0, 1);
                    }
                    
                    double prediction = simulator.simulateInference(features);
                    if (j == 0) {
                        logger.info("   Sample prediction: {:.4f}", prediction);
                    }
                }
                
                // Get performance statistics
                Map<String, Double> stats = simulator.getPerformanceStats();
                logger.info("   Model size: {}MB", String.format("%.2f", mobileModel.getModelSizeMB()));
                logger.info("   Avg inference time: {}ms", String.format("%.2f", stats.get("avg_inference_time_ms")));
                logger.info("   Max memory usage: {}MB", String.format("%.2f", stats.get("max_memory_usage_mb")));
                logger.info("   Meets constraints: {}", simulator.meetsPerformanceConstraints());
                
            } catch (Exception e) {
                logger.error("   Deployment failed: {}", e.getMessage());
            }
        }
        
        // Demonstrate batch inference
        logger.info("\nBatch Inference Demonstration:");
        MobileAIFramework.MobileModel testModel = MobileAIFramework.MobileDeploymentManager.deployModel(
            weights, biases, MobileAIFramework.DeviceProfiles.HIGH_END_PHONE);
        
        List<double[]> batchFeatures = new ArrayList<>();
        RandomDataGenerator random = new RandomDataGenerator();
        for (int i = 0; i < 5; i++) {
            double[] features = new double[weights.getColumnDimension()];
            for (int j = 0; j < features.length; j++) {
                features[j] = random.nextGaussian(0, 1);
            }
            batchFeatures.add(features);
        }
        
        double[] batchPredictions = testModel.predictBatch(batchFeatures);
        logger.info("   Batch predictions: {}", Arrays.toString(batchPredictions));
    }
    
    /**
     * Generate a sample model for demonstration
     */
    private static RealMatrix generateSampleModel() {
        int rows = 3;
        int cols = 10;
        
        double[][] weightsArray = MatrixUtils.randomNormal(rows, cols, 0.0, 0.1, 42L);
        RealMatrix weights = new Array2DRowRealMatrix(weightsArray);
        
        return weights;
    }
    
    /**
     * Utility method to print matrix statistics
     */
    private static void printMatrixStats(String name, RealMatrix matrix) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        int nonZeros = 0;
        
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double value = matrix.getEntry(i, j);
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
                if (Math.abs(value) > 1e-10) {
                    nonZeros++;
                }
            }
        }
        
        double mean = sum / (matrix.getRowDimension() * matrix.getColumnDimension());
        double sparsity = 1.0 - (double) nonZeros / (matrix.getRowDimension() * matrix.getColumnDimension());
        
        logger.info("{}: {}x{}, min={}, max={}, mean={}, sparsity={}%", 
                   name, matrix.getRowDimension(), matrix.getColumnDimension(), 
                   String.format("%.4f", min), String.format("%.4f", max), String.format("%.4f", mean), 
                   String.format("%.2f", sparsity * 100));
    }
}

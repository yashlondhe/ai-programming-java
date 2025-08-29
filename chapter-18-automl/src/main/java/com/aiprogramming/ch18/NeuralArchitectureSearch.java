package com.aiprogramming.ch18;

import java.util.*;

/**
 * Neural Architecture Search (NAS) implementation
 * Automatically discovers optimal neural network architectures
 */
public class NeuralArchitectureSearch {
    
    private final NASConfig config;
    private final Random random;
    
    public NeuralArchitectureSearch(NASConfig config) {
        this.config = config;
        this.random = new Random(config.getRandomSeed());
    }
    
    /**
     * Search for optimal neural network architecture
     */
    public NASResult search(double[][] features, double[] targets) {
        System.out.println("Starting Neural Architecture Search...");
        
        List<ArchitectureTrial> trials = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;
        NeuralArchitecture bestArchitecture = null;
        
        for (int trial = 0; trial < config.getMaxTrials(); trial++) {
            System.out.printf("Trial %d/%d%n", trial + 1, config.getMaxTrials());
            
            // Generate random architecture
            NeuralArchitecture architecture = generateRandomArchitecture();
            
            // Train and evaluate architecture
            double score = evaluateArchitecture(architecture, features, targets);
            
            trials.add(new ArchitectureTrial(architecture, score));
            
            if (score > bestScore) {
                bestScore = score;
                bestArchitecture = architecture;
                System.out.printf("New best architecture found! Score: %.4f%n", score);
            }
        }
        
        return new NASResult(bestArchitecture, bestScore, trials);
    }
    
    /**
     * Generate a random neural network architecture
     */
    private NeuralArchitecture generateRandomArchitecture() {
        int numLayers = random.nextInt(config.getMaxLayers() - config.getMinLayers() + 1) + config.getMinLayers();
        List<LayerConfig> layers = new ArrayList<>();
        
        int currentSize = config.getInputSize();
        
        for (int i = 0; i < numLayers; i++) {
            // Random layer size
            int layerSize = random.nextInt(config.getMaxLayerSize() - config.getMinLayerSize() + 1) + config.getMinLayerSize();
            
            // Random activation function
            String activation = config.getActivationFunctions()[random.nextInt(config.getActivationFunctions().length)];
            
            // Random dropout rate
            double dropout = random.nextDouble() * config.getMaxDropout();
            
            layers.add(new LayerConfig(layerSize, activation, dropout));
            currentSize = layerSize;
        }
        
        return new NeuralArchitecture(layers, config.getOutputSize());
    }
    
    /**
     * Evaluate a neural network architecture
     */
    private double evaluateArchitecture(NeuralArchitecture architecture, double[][] features, double[] targets) {
        try {
            // Create neural network from architecture
            NeuralNetwork network = createNetworkFromArchitecture(architecture);
            
            // Train the network
            network.train(features, targets);
            
            // Evaluate performance
            double score = network.evaluate(features, targets);
            
            // Check for invalid scores
            if (Double.isNaN(score) || Double.isInfinite(score)) {
                return Double.NEGATIVE_INFINITY;
            }
            
            return score;
        } catch (Exception e) {
            System.err.println("Error evaluating architecture: " + e.getMessage());
            return Double.NEGATIVE_INFINITY;
        }
    }
    
    /**
     * Create a neural network from architecture specification
     */
    private NeuralNetwork createNetworkFromArchitecture(NeuralArchitecture architecture) {
        NeuralNetwork network = new NeuralNetwork();
        
        // Set hyperparameters based on architecture
        Map<String, Object> hyperparameters = new HashMap<>();
        hyperparameters.put("learningRate", 0.01);
        hyperparameters.put("hiddenLayers", architecture.getLayers().size());
        hyperparameters.put("neuronsPerLayer", architecture.getLayers().get(0).getSize());
        hyperparameters.put("dropout", architecture.getLayers().get(0).getDropout());
        
        network.setHyperparameters(hyperparameters);
        
        return network;
    }
    
    /**
     * Neural network architecture specification
     */
    public static class NeuralArchitecture {
        private final List<LayerConfig> layers;
        private final int outputSize;
        
        public NeuralArchitecture(List<LayerConfig> layers, int outputSize) {
            this.layers = layers;
            this.outputSize = outputSize;
        }
        
        public List<LayerConfig> getLayers() {
            return layers;
        }
        
        public int getOutputSize() {
            return outputSize;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("NeuralArchitecture{layers=[");
            for (int i = 0; i < layers.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(layers.get(i));
            }
            sb.append("], outputSize=").append(outputSize).append("}");
            return sb.toString();
        }
    }
    
    /**
     * Layer configuration
     */
    public static class LayerConfig {
        private final int size;
        private final String activation;
        private final double dropout;
        
        public LayerConfig(int size, String activation, double dropout) {
            this.size = size;
            this.activation = activation;
            this.dropout = dropout;
        }
        
        public int getSize() {
            return size;
        }
        
        public String getActivation() {
            return activation;
        }
        
        public double getDropout() {
            return dropout;
        }
        
        @Override
        public String toString() {
            return String.format("Layer(size=%d, activation=%s, dropout=%.2f)", size, activation, dropout);
        }
    }
    
    /**
     * Architecture trial result
     */
    public static class ArchitectureTrial {
        private final NeuralArchitecture architecture;
        private final double score;
        
        public ArchitectureTrial(NeuralArchitecture architecture, double score) {
            this.architecture = architecture;
            this.score = score;
        }
        
        public NeuralArchitecture getArchitecture() {
            return architecture;
        }
        
        public double getScore() {
            return score;
        }
        
        @Override
        public String toString() {
            return String.format("ArchitectureTrial{score=%.4f, architecture=%s}", score, architecture);
        }
    }
    
    /**
     * NAS result
     */
    public static class NASResult {
        private final NeuralArchitecture bestArchitecture;
        private final double bestScore;
        private final List<ArchitectureTrial> trials;
        
        public NASResult(NeuralArchitecture bestArchitecture, double bestScore, List<ArchitectureTrial> trials) {
            this.bestArchitecture = bestArchitecture;
            this.bestScore = bestScore;
            this.trials = trials;
        }
        
        public NeuralArchitecture getBestArchitecture() {
            return bestArchitecture;
        }
        
        public double getBestScore() {
            return bestScore;
        }
        
        public List<ArchitectureTrial> getTrials() {
            return trials;
        }
        
        public void printSummary() {
            System.out.println("\n=== Neural Architecture Search Results ===");
            System.out.println("Best Score: " + String.format("%.4f", bestScore));
            System.out.println("Best Architecture: " + bestArchitecture);
            System.out.println("Total Trials: " + trials.size());
            
            // Show top 5 architectures
            System.out.println("\nTop 5 Architectures:");
            trials.stream()
                  .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
                  .limit(5)
                  .forEach(trial -> {
                      System.out.printf("Score: %.4f, Architecture: %s%n", 
                                      trial.getScore(), trial.getArchitecture());
                  });
        }
    }
}

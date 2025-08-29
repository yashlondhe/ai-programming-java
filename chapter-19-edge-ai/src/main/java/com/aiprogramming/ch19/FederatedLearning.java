package com.aiprogramming.ch19;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Federated Learning implementation for edge AI
 * Enables privacy-preserving model training across distributed devices
 */
public class FederatedLearning {
    
    private static final Logger logger = LoggerFactory.getLogger(FederatedLearning.class);
    
    /**
     * Federated learning client
     */
    public static class FederatedClient {
        private final String clientId;
        private final RealMatrix localWeights;
        private final double[] localBiases;
        private final List<double[]> localData;
        private final int localDataSize;
        
        public FederatedClient(String clientId, RealMatrix initialWeights, double[] initialBiases, List<double[]> localData) {
            this.clientId = clientId;
            this.localWeights = initialWeights.copy();
            this.localBiases = initialBiases.clone();
            this.localData = localData;
            this.localDataSize = localData.size();
        }
        
        /**
         * Train model on local data
         */
        public void trainLocalModel(int epochs, double learningRate) {
            logger.info("Client {} training for {} epochs", clientId, epochs);
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalLoss = 0.0;
                
                for (double[] sample : localData) {
                    // Simple gradient descent update
                    double[] features = Arrays.copyOf(sample, sample.length - 1);
                    double target = sample[sample.length - 1];
                    
                    // Forward pass
                    double prediction = predict(features);
                    double loss = 0.5 * Math.pow(prediction - target, 2);
                    totalLoss += loss;
                    
                    // Backward pass (simplified)
                    double gradient = prediction - target;
                    
                    // Update weights
                    for (int i = 0; i < localWeights.getRowDimension(); i++) {
                        for (int j = 0; j < localWeights.getColumnDimension(); j++) {
                            double currentWeight = localWeights.getEntry(i, j);
                            double weightGradient = gradient * features[j % features.length];
                            localWeights.setEntry(i, j, currentWeight - learningRate * weightGradient);
                        }
                    }
                    
                    // Update biases
                    for (int i = 0; i < localBiases.length; i++) {
                        localBiases[i] -= learningRate * gradient;
                    }
                }
                
                if (epoch % 10 == 0) {
                    logger.debug("Client {} epoch {}: loss = {:.4f}", clientId, epoch, totalLoss / localDataSize);
                }
            }
        }
        
        /**
         * Make prediction using local model
         */
        public double predict(double[] features) {
            double result = 0.0;
            
            for (int i = 0; i < localWeights.getRowDimension(); i++) {
                for (int j = 0; j < localWeights.getColumnDimension(); j++) {
                    result += localWeights.getEntry(i, j) * features[j % features.length];
                }
                result += localBiases[i];
            }
            
            return result;
        }
        
        /**
         * Get model parameters for aggregation
         */
        public ModelParameters getModelParameters() {
            return new ModelParameters(localWeights.copy(), localBiases.clone(), localDataSize);
        }
        
        /**
         * Update local model with aggregated parameters
         */
        public void updateModel(RealMatrix newWeights, double[] newBiases) {
            for (int i = 0; i < localWeights.getRowDimension(); i++) {
                for (int j = 0; j < localWeights.getColumnDimension(); j++) {
                    localWeights.setEntry(i, j, newWeights.getEntry(i, j));
                }
            }
            
            System.arraycopy(newBiases, 0, localBiases, 0, localBiases.length);
        }
        
        public String getClientId() { return clientId; }
        public int getLocalDataSize() { return localDataSize; }
    }
    
    /**
     * Model parameters container
     */
    public static class ModelParameters {
        private final RealMatrix weights;
        private final double[] biases;
        private final int dataSize;
        
        public ModelParameters(RealMatrix weights, double[] biases, int dataSize) {
            this.weights = weights;
            this.biases = biases;
            this.dataSize = dataSize;
        }
        
        public RealMatrix getWeights() { return weights; }
        public double[] getBiases() { return biases; }
        public int getDataSize() { return dataSize; }
    }
    
    /**
     * Federated learning server
     */
    public static class FederatedServer {
        private final List<FederatedClient> clients;
        private final RealMatrix globalWeights;
        private final double[] globalBiases;
        private final int totalRounds;
        private final int localEpochs;
        private final double learningRate;
        
        public FederatedServer(List<FederatedClient> clients, RealMatrix initialWeights, 
                             double[] initialBiases, int totalRounds, int localEpochs, double learningRate) {
            this.clients = clients;
            this.globalWeights = initialWeights.copy();
            this.globalBiases = initialBiases.clone();
            this.totalRounds = totalRounds;
            this.localEpochs = localEpochs;
            this.learningRate = learningRate;
        }
        
        /**
         * Run federated learning training
         */
        public void train() {
            logger.info("Starting federated learning with {} clients for {} rounds", clients.size(), totalRounds);
            
            for (int round = 0; round < totalRounds; round++) {
                logger.info("=== Federated Round {} ===", round + 1);
                
                // Step 1: Distribute global model to clients
                distributeGlobalModel();
                
                // Step 2: Train on local data (parallel)
                trainClientsInParallel();
                
                // Step 3: Aggregate local models
                aggregateModels();
                
                // Step 4: Evaluate global model
                evaluateGlobalModel(round);
            }
        }
        
        /**
         * Distribute global model to all clients
         */
        private void distributeGlobalModel() {
            for (FederatedClient client : clients) {
                client.updateModel(globalWeights, globalBiases);
            }
        }
        
        /**
         * Train all clients in parallel
         */
        private void trainClientsInParallel() {
            ExecutorService executor = Executors.newFixedThreadPool(clients.size());
            List<Future<Void>> futures = new ArrayList<>();
            
            for (FederatedClient client : clients) {
                Future<Void> future = executor.submit(() -> {
                    client.trainLocalModel(localEpochs, learningRate);
                    return null;
                });
                futures.add(future);
            }
            
            // Wait for all clients to finish training
            for (Future<Void> future : futures) {
                try {
                    future.get();
                } catch (InterruptedException | ExecutionException e) {
                    logger.error("Error in client training", e);
                }
            }
            
            executor.shutdown();
        }
        
        /**
         * Aggregate local models using FedAvg algorithm
         */
        private void aggregateModels() {
            logger.info("Aggregating models from {} clients", clients.size());
            
            // Collect model parameters from all clients
            List<ModelParameters> modelParameters = clients.stream()
                .map(FederatedClient::getModelParameters)
                .collect(Collectors.toList());
            
            // Calculate total data size
            int totalDataSize = modelParameters.stream()
                .mapToInt(ModelParameters::getDataSize)
                .sum();
            
            // Initialize aggregated weights and biases
            int rows = globalWeights.getRowDimension();
            int cols = globalWeights.getColumnDimension();
            RealMatrix aggregatedWeights = new Array2DRowRealMatrix(rows, cols);
            double[] aggregatedBiases = new double[globalBiases.length];
            
            // Weighted average aggregation (FedAvg)
            for (ModelParameters params : modelParameters) {
                double weight = (double) params.getDataSize() / totalDataSize;
                
                // Aggregate weights
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        double currentValue = aggregatedWeights.getEntry(i, j);
                        double clientValue = params.getWeights().getEntry(i, j);
                        aggregatedWeights.setEntry(i, j, currentValue + weight * clientValue);
                    }
                }
                
                // Aggregate biases
                for (int i = 0; i < aggregatedBiases.length; i++) {
                    aggregatedBiases[i] += weight * params.getBiases()[i];
                }
            }
            
            // Update global model
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    globalWeights.setEntry(i, j, aggregatedWeights.getEntry(i, j));
                }
            }
            
            System.arraycopy(aggregatedBiases, 0, globalBiases, 0, globalBiases.length);
            
            logger.info("Model aggregation completed");
        }
        
        /**
         * Evaluate global model performance
         */
        private void evaluateGlobalModel(int round) {
            // Simple evaluation on a subset of data
            double totalLoss = 0.0;
            int sampleCount = 0;
            
            for (FederatedClient client : clients) {
                List<double[]> clientData = client.localData;
                int evalSize = Math.min(10, clientData.size()); // Evaluate on 10 samples per client
                
                for (int i = 0; i < evalSize; i++) {
                    double[] sample = clientData.get(i);
                    double[] features = Arrays.copyOf(sample, sample.length - 1);
                    double target = sample[sample.length - 1];
                    
                    double prediction = predict(features);
                    double loss = 0.5 * Math.pow(prediction - target, 2);
                    totalLoss += loss;
                    sampleCount++;
                }
            }
            
            double avgLoss = totalLoss / sampleCount;
            logger.info("Round {} - Global model average loss: {:.4f}", round + 1, avgLoss);
        }
        
        /**
         * Make prediction using global model
         */
        public double predict(double[] features) {
            double result = 0.0;
            
            for (int i = 0; i < globalWeights.getRowDimension(); i++) {
                for (int j = 0; j < globalWeights.getColumnDimension(); j++) {
                    result += globalWeights.getEntry(i, j) * features[j % features.length];
                }
                result += globalBiases[i];
            }
            
            return result;
        }
        
        public RealMatrix getGlobalWeights() { return globalWeights.copy(); }
        public double[] getGlobalBiases() { return globalBiases.clone(); }
    }
    
    /**
     * Generate synthetic federated learning data
     */
    public static List<double[]> generateSyntheticData(int numSamples, int numFeatures, double noiseLevel) {
        RandomDataGenerator random = new RandomDataGenerator();
        List<double[]> data = new ArrayList<>();
        
        // Generate random weights for the true model
        double[] trueWeights = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            trueWeights[i] = random.nextGaussian(0, 1);
        }
        double trueBias = random.nextGaussian(0, 1);
        
        for (int i = 0; i < numSamples; i++) {
            double[] features = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                features[j] = random.nextGaussian(0, 1);
            }
            
            // Generate target using linear model + noise
            double target = trueBias;
            for (int j = 0; j < numFeatures; j++) {
                target += trueWeights[j] * features[j];
            }
            target += random.nextGaussian(0, noiseLevel);
            
            // Create sample with features + target
            double[] sample = new double[numFeatures + 1];
            System.arraycopy(features, 0, sample, 0, numFeatures);
            sample[numFeatures] = target;
            
            data.add(sample);
        }
        
        return data;
    }
    
    /**
     * Create federated learning clients with distributed data
     */
    public static List<FederatedClient> createClients(int numClients, int samplesPerClient, 
                                                    int numFeatures, RealMatrix initialWeights, 
                                                    double[] initialBiases) {
        List<FederatedClient> clients = new ArrayList<>();
        
        for (int i = 0; i < numClients; i++) {
            // Generate local data for each client
            List<double[]> localData = generateSyntheticData(samplesPerClient, numFeatures, 0.1);
            
            FederatedClient client = new FederatedClient("client_" + i, initialWeights, initialBiases, localData);
            clients.add(client);
        }
        
        return clients;
    }
    
    /**
     * Run federated learning experiment
     */
    public static FederatedServer runFederatedLearningExperiment(int numClients, int samplesPerClient, 
                                                                int numFeatures, int totalRounds, 
                                                                int localEpochs, double learningRate) {
        logger.info("Starting federated learning experiment");
        logger.info("Clients: {}, Samples per client: {}, Features: {}", numClients, samplesPerClient, numFeatures);
        logger.info("Rounds: {}, Local epochs: {}, Learning rate: {}", totalRounds, localEpochs, learningRate);
        
        // Initialize global model
        RealMatrix initialWeights = new Array2DRowRealMatrix(1, numFeatures);
        double[] initialBiases = new double[1];
        
        // Create clients
        List<FederatedClient> clients = createClients(numClients, samplesPerClient, numFeatures, 
                                                    initialWeights, initialBiases);
        
        // Create and run federated server
        FederatedServer server = new FederatedServer(clients, initialWeights, initialBiases, 
                                                    totalRounds, localEpochs, learningRate);
        server.train();
        
        return server;
    }
}

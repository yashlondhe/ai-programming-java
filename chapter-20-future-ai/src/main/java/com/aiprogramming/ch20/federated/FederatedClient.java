package com.aiprogramming.ch20.federated;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Federated Learning Client
 * 
 * Implements a federated learning client that can train models locally
 * while preserving data privacy. The client participates in federated
 * learning rounds by training on local data and contributing model updates
 * to a global model without sharing raw data.
 */
public class FederatedClient {
    
    private static final Logger logger = LoggerFactory.getLogger(FederatedClient.class);
    
    private final String clientId;
    private final List<DataPoint> localData;
    private final FederatedModel localModel;
    private final PrivacyPreservingML privacyProcessor;
    private final RandomGenerator random;
    
    private int roundNumber;
    private double learningRate;
    private int localEpochs;
    private boolean differentialPrivacyEnabled;
    private double privacyEpsilon;
    
    public FederatedClient(String clientId, List<DataPoint> localData) {
        this.clientId = clientId;
        this.localData = new ArrayList<>(localData);
        this.localModel = new FederatedModel();
        this.privacyProcessor = new PrivacyPreservingML();
        this.random = new Well19937c();
        
        this.roundNumber = 0;
        this.learningRate = 0.01;
        this.localEpochs = 5;
        this.differentialPrivacyEnabled = true;
        this.privacyEpsilon = 1.0;
    }
    
    /**
     * Represents a data point with features and label
     */
    public static class DataPoint {
        private final RealVector features;
        private final double label;
        
        public DataPoint(double[] features, double label) {
            this.features = new ArrayRealVector(features);
            this.label = label;
        }
        
        public RealVector getFeatures() {
            return features;
        }
        
        public double getLabel() {
            return label;
        }
    }
    
    /**
     * Represents a federated model with parameters
     */
    public static class FederatedModel {
        private RealVector weights;
        private double bias;
        private int featureDimension;
        
        public FederatedModel() {
            this.featureDimension = 0;
        }
        
        public void initialize(int featureDimension) {
            this.featureDimension = featureDimension;
            this.weights = new ArrayRealVector(featureDimension);
            this.bias = 0.0;
            
            // Initialize with small random values
            for (int i = 0; i < featureDimension; i++) {
                weights.setEntry(i, (Math.random() - 0.5) * 0.1);
            }
        }
        
        public double predict(RealVector features) {
            return weights.dotProduct(features) + bias;
        }
        
        public RealVector getWeights() {
            return weights;
        }
        
        public void setWeights(RealVector weights) {
            this.weights = weights;
        }
        
        public double getBias() {
            return bias;
        }
        
        public void setBias(double bias) {
            this.bias = bias;
        }
        
        public int getFeatureDimension() {
            return featureDimension;
        }
    }
    
    /**
     * Train the local model on local data
     * 
     * @param globalModel The current global model parameters
     * @return Model update to be sent to the server
     */
    public ModelUpdate train(FederatedModel globalModel) {
        logger.info("Client {} starting local training for round {}", clientId, roundNumber);
        
        // Initialize local model if needed
        if (localModel.getFeatureDimension() == 0) {
            localModel.initialize(globalModel.getFeatureDimension());
        }
        
        // Copy global model parameters to local model
        localModel.setWeights(globalModel.getWeights().copy());
        localModel.setBias(globalModel.getBias());
        
        // Local training epochs
        for (int epoch = 0; epoch < localEpochs; epoch++) {
            trainEpoch();
        }
        
        // Calculate model update (difference between local and global model)
        RealVector weightUpdate = localModel.getWeights().subtract(globalModel.getWeights());
        double biasUpdate = localModel.getBias() - globalModel.getBias();
        
        // Apply privacy-preserving techniques
        if (differentialPrivacyEnabled) {
            weightUpdate = privacyProcessor.addDifferentialPrivacy(weightUpdate, privacyEpsilon);
            biasUpdate = privacyProcessor.addDifferentialPrivacy(biasUpdate, privacyEpsilon);
        }
        
        // Create model update
        ModelUpdate update = new ModelUpdate(clientId, roundNumber, weightUpdate, biasUpdate, localData.size());
        
        roundNumber++;
        logger.info("Client {} completed training for round {}", clientId, roundNumber - 1);
        
        return update;
    }
    
    /**
     * Train for one epoch on local data
     */
    private void trainEpoch() {
        // Shuffle data for better training
        List<DataPoint> shuffledData = new ArrayList<>(localData);
        java.util.Collections.shuffle(shuffledData, new java.util.Random(random.nextLong()));
        
        for (DataPoint dataPoint : shuffledData) {
            // Forward pass
            double prediction = localModel.predict(dataPoint.getFeatures());
            double error = prediction - dataPoint.getLabel();
            
            // Backward pass (gradient descent)
            RealVector gradients = dataPoint.getFeatures().mapMultiply(error);
            
            // Update weights
            RealVector newWeights = localModel.getWeights().subtract(gradients.mapMultiply(learningRate));
            localModel.setWeights(newWeights);
            
            // Update bias
            double newBias = localModel.getBias() - learningRate * error;
            localModel.setBias(newBias);
        }
    }
    
    /**
     * Evaluate local model performance
     * 
     * @return Evaluation metrics
     */
    public Map<String, Double> evaluate() {
        if (localData.isEmpty()) {
            return new HashMap<>();
        }
        
        double totalLoss = 0.0;
        double totalAccuracy = 0.0;
        
        for (DataPoint dataPoint : localData) {
            double prediction = localModel.predict(dataPoint.getFeatures());
            double error = prediction - dataPoint.getLabel();
            totalLoss += error * error;
            
            // For binary classification (assuming threshold at 0.5)
            if (dataPoint.getLabel() >= 0.5) {
                totalAccuracy += (prediction >= 0.5) ? 1.0 : 0.0;
            } else {
                totalAccuracy += (prediction < 0.5) ? 1.0 : 0.0;
            }
        }
        
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("loss", totalLoss / localData.size());
        metrics.put("accuracy", totalAccuracy / localData.size());
        metrics.put("dataSize", (double) localData.size());
        
        return metrics;
    }
    
    /**
     * Represents a model update to be sent to the federated server
     */
    public static class ModelUpdate {
        private final String clientId;
        private final int roundNumber;
        private final RealVector weightUpdate;
        private final double biasUpdate;
        private final int dataSize;
        private final long timestamp;
        
        public ModelUpdate(String clientId, int roundNumber, RealVector weightUpdate, 
                          double biasUpdate, int dataSize) {
            this.clientId = clientId;
            this.roundNumber = roundNumber;
            this.weightUpdate = weightUpdate;
            this.biasUpdate = biasUpdate;
            this.dataSize = dataSize;
            this.timestamp = System.currentTimeMillis();
        }
        
        // Getters
        public String getClientId() { return clientId; }
        public int getRoundNumber() { return roundNumber; }
        public RealVector getWeightUpdate() { return weightUpdate; }
        public double getBiasUpdate() { return biasUpdate; }
        public int getDataSize() { return dataSize; }
        public long getTimestamp() { return timestamp; }
    }
    
    /**
     * Generate synthetic data for testing
     */
    public static List<DataPoint> generateSyntheticData(int numSamples, int featureDimension, 
                                                       String clientId) {
        List<DataPoint> data = new ArrayList<>();
        RandomGenerator random = new Well19937c(clientId.hashCode());
        
        // Generate synthetic linear data with some noise
        double[] trueWeights = new double[featureDimension];
        for (int i = 0; i < featureDimension; i++) {
            trueWeights[i] = (random.nextDouble() - 0.5) * 2.0;
        }
        double trueBias = (random.nextDouble() - 0.5) * 2.0;
        
        for (int i = 0; i < numSamples; i++) {
            double[] features = new double[featureDimension];
            for (int j = 0; j < featureDimension; j++) {
                features[j] = random.nextDouble() * 2.0 - 1.0;
            }
            
            // Calculate true label
            double trueValue = 0.0;
            for (int j = 0; j < featureDimension; j++) {
                trueValue += trueWeights[j] * features[j];
            }
            trueValue += trueBias;
            
            // Add noise and convert to binary classification
            double noise = random.nextGaussian() * 0.1;
            double label = (trueValue + noise > 0) ? 1.0 : 0.0;
            
            data.add(new DataPoint(features, label));
        }
        
        return data;
    }
    
    // Configuration methods
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public void setLocalEpochs(int localEpochs) {
        this.localEpochs = localEpochs;
    }
    
    public void setDifferentialPrivacyEnabled(boolean enabled) {
        this.differentialPrivacyEnabled = enabled;
    }
    
    public void setPrivacyEpsilon(double epsilon) {
        this.privacyEpsilon = epsilon;
    }
    
    public String getClientId() {
        return clientId;
    }
    
    public int getDataSize() {
        return localData.size();
    }
}

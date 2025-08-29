package com.aiprogramming.ch18;

import java.util.*;

/**
 * Simple Random Forest model
 */
public class RandomForest implements MLModel {
    
    private List<DecisionTree> trees;
    private Map<String, Object> hyperparameters;
    private double[] featureImportance;
    
    public RandomForest() {
        this.hyperparameters = new HashMap<>();
        this.hyperparameters.put("numTrees", 10);
        this.hyperparameters.put("maxDepth", 5);
        this.hyperparameters.put("minSamplesSplit", 2);
        this.trees = new ArrayList<>();
    }
    
    @Override
    public void train(double[][] features, double[] targets) {
        int numTrees = (Integer) hyperparameters.get("numTrees");
        int maxDepth = (Integer) hyperparameters.get("maxDepth");
        int minSamplesSplit = (Integer) hyperparameters.get("minSamplesSplit");
        
        trees.clear();
        
        // Train multiple trees
        for (int i = 0; i < numTrees; i++) {
            DecisionTree tree = new DecisionTree(maxDepth, minSamplesSplit);
            
            // Bootstrap sample
            int[] bootstrapIndices = generateBootstrapSample(features.length);
            double[][] bootstrapFeatures = new double[bootstrapIndices.length][features[0].length];
            double[] bootstrapTargets = new double[bootstrapIndices.length];
            
            for (int j = 0; j < bootstrapIndices.length; j++) {
                bootstrapFeatures[j] = features[bootstrapIndices[j]];
                bootstrapTargets[j] = targets[bootstrapIndices[j]];
            }
            
            tree.train(bootstrapFeatures, bootstrapTargets);
            trees.add(tree);
        }
        
        // Calculate feature importance
        calculateFeatureImportance(features, targets);
    }
    
    @Override
    public double[] predict(double[][] features) {
        double[] predictions = new double[features.length];
        
        for (int i = 0; i < features.length; i++) {
            double sum = 0.0;
            for (DecisionTree tree : trees) {
                sum += tree.predict(features[i]);
            }
            predictions[i] = sum / trees.size();
        }
        
        return predictions;
    }
    
    @Override
    public double evaluate(double[][] features, double[] targets) {
        double[] predictions = predict(features);
        double mse = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - targets[i];
            mse += error * error;
        }
        return mse / predictions.length;
    }
    
    @Override
    public void setHyperparameters(Map<String, Object> hyperparameters) {
        this.hyperparameters.putAll(hyperparameters);
    }
    
    @Override
    public Map<String, Object> getHyperparameters() {
        return new HashMap<>(hyperparameters);
    }
    
    public double[] getFeatureImportance() {
        return featureImportance;
    }
    
    private int[] generateBootstrapSample(int size) {
        Random random = new Random();
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = random.nextInt(size);
        }
        return indices;
    }
    
            private void calculateFeatureImportance(double[][] features, double[] targets) {
            int numFeatures = features[0].length;
            featureImportance = new double[numFeatures];
            
            // Simple feature importance based on tree splits
            for (DecisionTree tree : trees) {
                double[] treeImportance = tree.getFeatureImportance();
                // Ensure we don't exceed the array bounds
                int maxIndex = Math.min(numFeatures, treeImportance.length);
                for (int i = 0; i < maxIndex; i++) {
                    featureImportance[i] += treeImportance[i];
                }
            }
            
            // Normalize
            double sum = Arrays.stream(featureImportance).sum();
            if (sum > 0) {
                for (int i = 0; i < numFeatures; i++) {
                    featureImportance[i] /= sum;
                }
            }
        }
    
    /**
     * Simple Decision Tree implementation
     */
    private static class DecisionTree {
        private Node root;
        private final int maxDepth;
        private final int minSamplesSplit;
        
        public DecisionTree(int maxDepth, int minSamplesSplit) {
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
        }
        
        public void train(double[][] features, double[] targets) {
            root = buildTree(features, targets, 0);
        }
        
        public double predict(double[] features) {
            return predictNode(root, features);
        }
        
        public double[] getFeatureImportance() {
            // Get the actual number of features from the training data
            // For now, use a reasonable default based on the node structure
            int numFeatures = 0;
            if (root != null) {
                // Find the maximum feature index used in the tree
                numFeatures = findMaxFeatureIndex(root) + 1;
            }
            double[] importance = new double[numFeatures];
            calculateImportance(root, importance);
            return importance;
        }
        
        private int findMaxFeatureIndex(Node node) {
            if (node == null) return -1;
            int maxIndex = node.featureIndex;
            if (node.children != null) {
                for (Node child : node.children) {
                    maxIndex = Math.max(maxIndex, findMaxFeatureIndex(child));
                }
            }
            return maxIndex;
        }
        
        private Node buildTree(double[][] features, double[] targets, int depth) {
            if (depth >= maxDepth || features.length < minSamplesSplit) {
                return new Node(null, -1, 0, calculateMean(targets));
            }
            
            // Find best split
            SplitInfo bestSplit = findBestSplit(features, targets);
            
            if (bestSplit == null) {
                return new Node(null, -1, 0, calculateMean(targets));
            }
            
            // Split data
            List<double[]> leftFeatures = new ArrayList<>();
            List<double[]> rightFeatures = new ArrayList<>();
            List<Double> leftTargets = new ArrayList<>();
            List<Double> rightTargets = new ArrayList<>();
            
            for (int i = 0; i < features.length; i++) {
                if (features[i][bestSplit.featureIndex] <= bestSplit.threshold) {
                    leftFeatures.add(features[i]);
                    leftTargets.add(targets[i]);
                } else {
                    rightFeatures.add(features[i]);
                    rightTargets.add(targets[i]);
                }
            }
            
            // Convert to arrays
            double[][] leftFeaturesArray = leftFeatures.toArray(new double[0][]);
            double[][] rightFeaturesArray = rightFeatures.toArray(new double[0][]);
            double[] leftTargetsArray = leftTargets.stream().mapToDouble(Double::doubleValue).toArray();
            double[] rightTargetsArray = rightTargets.stream().mapToDouble(Double::doubleValue).toArray();
            
            // Build children
            Node leftChild = buildTree(leftFeaturesArray, leftTargetsArray, depth + 1);
            Node rightChild = buildTree(rightFeaturesArray, rightTargetsArray, depth + 1);
            
            return new Node(new Node[]{leftChild, rightChild}, bestSplit.featureIndex, 
                          bestSplit.threshold, 0);
        }
        
        private SplitInfo findBestSplit(double[][] features, double[] targets) {
            int numFeatures = features[0].length;
            SplitInfo bestSplit = null;
            double bestGain = 0;
            
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                double[] featureValues = new double[features.length];
                for (int i = 0; i < features.length; i++) {
                    featureValues[i] = features[i][featureIndex];
                }
                
                // Try different thresholds
                for (int i = 0; i < features.length; i++) {
                    double threshold = featureValues[i];
                    
                    List<Double> leftTargets = new ArrayList<>();
                    List<Double> rightTargets = new ArrayList<>();
                    
                    for (int j = 0; j < features.length; j++) {
                        if (features[j][featureIndex] <= threshold) {
                            leftTargets.add(targets[j]);
                        } else {
                            rightTargets.add(targets[j]);
                        }
                    }
                    
                    if (leftTargets.size() > 0 && rightTargets.size() > 0) {
                        double gain = calculateInformationGain(targets, leftTargets, rightTargets);
                        if (gain > bestGain) {
                            bestGain = gain;
                            bestSplit = new SplitInfo(featureIndex, threshold);
                        }
                    }
                }
            }
            
            return bestSplit;
        }
        
        private double calculateInformationGain(double[] targets, List<Double> leftTargets, 
                                              List<Double> rightTargets) {
            double parentEntropy = calculateEntropy(targets);
            double leftEntropy = calculateEntropy(leftTargets.stream().mapToDouble(Double::doubleValue).toArray());
            double rightEntropy = calculateEntropy(rightTargets.stream().mapToDouble(Double::doubleValue).toArray());
            
            double leftWeight = (double) leftTargets.size() / targets.length;
            double rightWeight = (double) rightTargets.size() / targets.length;
            
            return parentEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);
        }
        
        private double calculateEntropy(double[] values) {
            // Simplified entropy calculation for regression
            double mean = calculateMean(values);
            double variance = 0;
            for (double value : values) {
                variance += (value - mean) * (value - mean);
            }
            variance /= values.length;
            return Math.log(variance + 1); // Add 1 to avoid log(0)
        }
        
        private double calculateMean(double[] values) {
            return Arrays.stream(values).average().orElse(0.0);
        }
        
        private double predictNode(Node node, double[] features) {
            if (node.children == null) {
                return node.value;
            }
            
            if (features[node.featureIndex] <= node.threshold) {
                return predictNode(node.children[0], features);
            } else {
                return predictNode(node.children[1], features);
            }
        }
        
        private void calculateImportance(Node node, double[] importance) {
            if (node == null || node.children == null) {
                return;
            }
            
            if (node.featureIndex >= 0) {
                importance[node.featureIndex]++;
            }
            
            for (Node child : node.children) {
                calculateImportance(child, importance);
            }
        }
        
        private static class Node {
            Node[] children;
            int featureIndex;
            double threshold;
            double value;
            
            public Node(Node[] children, int featureIndex, double threshold, double value) {
                this.children = children;
                this.featureIndex = featureIndex;
                this.threshold = threshold;
                this.value = value;
            }
        }
        
        private static class SplitInfo {
            int featureIndex;
            double threshold;
            
            public SplitInfo(int featureIndex, double threshold) {
                this.featureIndex = featureIndex;
                this.threshold = threshold;
            }
        }
    }
}

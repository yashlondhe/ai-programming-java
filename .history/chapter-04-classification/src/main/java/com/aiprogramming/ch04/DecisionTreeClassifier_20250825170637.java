package com.aiprogramming.ch04;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Decision Tree Classifier
 * 
 * A tree-based classification algorithm that recursively splits the data
 * based on feature values to create a hierarchical decision structure.
 */
public class DecisionTreeClassifier implements Classifier {
    
    private final int maxDepth;
    private TreeNode root;
    private Set<String> featureNames;
    
    public DecisionTreeClassifier(int maxDepth) {
        this.maxDepth = maxDepth;
        this.root = null;
        this.featureNames = new HashSet<>();
    }
    
    @Override
    public void train(List<ClassificationDataPoint> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract feature names
        this.featureNames.clear();
        for (ClassificationDataPoint point : trainingData) {
            featureNames.addAll(point.getFeatures().keySet());
        }
        
        // Build the decision tree
        this.root = buildTree(trainingData, 0);
    }
    
    @Override
    public String predict(Map<String, Double> features) {
        if (root == null) {
            throw new IllegalStateException("Classifier must be trained before making predictions");
        }
        
        return predictRecursive(root, features);
    }
    
    /**
     * Recursively builds the decision tree
     */
    private TreeNode buildTree(List<ClassificationDataPoint> data, int depth) {
        // Base cases
        if (data.isEmpty()) {
            return null;
        }
        
        if (depth >= maxDepth) {
            return new TreeNode(getMajorityClass(data), null, null, null, null, true);
        }
        
        // Check if all data points have the same class
        Set<String> uniqueClasses = data.stream()
                .map(ClassificationDataPoint::getLabel)
                .collect(Collectors.toSet());
        
        if (uniqueClasses.size() == 1) {
            return new TreeNode(uniqueClasses.iterator().next(), null, null, null, null, true);
        }
        
        // Find best split
        SplitInfo bestSplit = findBestSplit(data);
        
        if (bestSplit == null) {
            return new TreeNode(getMajorityClass(data), null, null, null, null, true);
        }
        
        // Split the data
        List<ClassificationDataPoint> leftData = new ArrayList<>();
        List<ClassificationDataPoint> rightData = new ArrayList<>();
        
        for (ClassificationDataPoint point : data) {
            double featureValue = point.getFeature(bestSplit.getFeatureName());
            if (featureValue <= bestSplit.getThreshold()) {
                leftData.add(point);
            } else {
                rightData.add(point);
            }
        }
        
        // Recursively build left and right subtrees
        TreeNode leftChild = buildTree(leftData, depth + 1);
        TreeNode rightChild = buildTree(rightData, depth + 1);
        
        return new TreeNode(null, bestSplit.getFeatureName(), bestSplit.getThreshold(), 
                          leftChild, rightChild, false);
    }
    
    /**
     * Finds the best split based on information gain
     */
    private SplitInfo findBestSplit(List<ClassificationDataPoint> data) {
        double bestInformationGain = -1;
        SplitInfo bestSplit = null;
        
        for (String featureName : featureNames) {
            List<Double> featureValues = data.stream()
                    .map(point -> point.getFeature(featureName))
                    .distinct()
                    .sorted()
                    .collect(Collectors.toList());
            
            // Try different thresholds
            for (int i = 0; i < featureValues.size() - 1; i++) {
                double threshold = (featureValues.get(i) + featureValues.get(i + 1)) / 2.0;
                
                double informationGain = calculateInformationGain(data, featureName, threshold);
                
                if (informationGain > bestInformationGain) {
                    bestInformationGain = informationGain;
                    bestSplit = new SplitInfo(featureName, threshold, informationGain);
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * Calculates information gain for a potential split
     */
    private double calculateInformationGain(List<ClassificationDataPoint> data, String featureName, double threshold) {
        double parentEntropy = calculateEntropy(data);
        
        List<ClassificationDataPoint> leftData = new ArrayList<>();
        List<ClassificationDataPoint> rightData = new ArrayList<>();
        
        for (ClassificationDataPoint point : data) {
            double featureValue = point.getFeature(featureName);
            if (featureValue <= threshold) {
                leftData.add(point);
            } else {
                rightData.add(point);
            }
        }
        
        double leftEntropy = calculateEntropy(leftData);
        double rightEntropy = calculateEntropy(rightData);
        
        double leftWeight = (double) leftData.size() / data.size();
        double rightWeight = (double) rightData.size() / data.size();
        
        return parentEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);
    }
    
    /**
     * Calculates entropy for a set of data points
     */
    private double calculateEntropy(List<ClassificationDataPoint> data) {
        if (data.isEmpty()) {
            return 0.0;
        }
        
        Map<String, Long> classCounts = data.stream()
                .collect(Collectors.groupingBy(
                    ClassificationDataPoint::getLabel,
                    Collectors.counting()
                ));
        
        double entropy = 0.0;
        int totalCount = data.size();
        
        for (long count : classCounts.values()) {
            double probability = (double) count / totalCount;
            if (probability > 0) {
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    /**
     * Gets the majority class from a set of data points
     */
    private String getMajorityClass(List<ClassificationDataPoint> data) {
        return data.stream()
                .collect(Collectors.groupingBy(
                    ClassificationDataPoint::getLabel,
                    Collectors.counting()
                ))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * Recursively predicts using the decision tree
     */
    private String predictRecursive(TreeNode node, Map<String, Double> features) {
        if (node.isLeaf()) {
            return node.getPredictedClass();
        }
        
        double featureValue = features.getOrDefault(node.getFeatureName(), 0.0);
        
        if (featureValue <= node.getThreshold()) {
            return predictRecursive(node.getLeftChild(), features);
        } else {
            return predictRecursive(node.getRightChild(), features);
        }
    }
    
    /**
     * Prints the decision tree structure
     */
    public void printTree() {
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }
        printTreeRecursive(root, 0);
    }
    
    private void printTreeRecursive(TreeNode node, int depth) {
        String indent = "  ".repeat(depth);
        
        if (node.isLeaf()) {
            System.out.println(indent + "Predict: " + node.getPredictedClass());
        } else {
            System.out.println(indent + "if " + node.getFeatureName() + " <= " + node.getThreshold());
            printTreeRecursive(node.getLeftChild(), depth + 1);
            System.out.println(indent + "else");
            printTreeRecursive(node.getRightChild(), depth + 1);
        }
    }
    
    /**
     * Represents a node in the decision tree
     */
    private static class TreeNode {
        private final String predictedClass;
        private final String featureName;
        private final Double threshold;
        private final TreeNode leftChild;
        private final TreeNode rightChild;
        private final boolean isLeaf;
        
        public TreeNode(String predictedClass, String featureName, Double threshold,
                       TreeNode leftChild, TreeNode rightChild, boolean isLeaf) {
            this.predictedClass = predictedClass;
            this.featureName = featureName;
            this.threshold = threshold;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
            this.isLeaf = isLeaf;
        }
        
        public String getPredictedClass() {
            return predictedClass;
        }
        
        public String getFeatureName() {
            return featureName;
        }
        
        public Double getThreshold() {
            return threshold;
        }
        
        public TreeNode getLeftChild() {
            return leftChild;
        }
        
        public TreeNode getRightChild() {
            return rightChild;
        }
        
        public boolean isLeaf() {
            return isLeaf;
        }
    }
    
    /**
     * Represents information about a split
     */
    private static class SplitInfo {
        private final String featureName;
        private final double threshold;
        private final double informationGain;
        
        public SplitInfo(String featureName, double threshold, double informationGain) {
            this.featureName = featureName;
            this.threshold = threshold;
            this.informationGain = informationGain;
        }
        
        public String getFeatureName() {
            return featureName;
        }
        
        public double getThreshold() {
            return threshold;
        }
        
        public double getInformationGain() {
            return informationGain;
        }
    }
}

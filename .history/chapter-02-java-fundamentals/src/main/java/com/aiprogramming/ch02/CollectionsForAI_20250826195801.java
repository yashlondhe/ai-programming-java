package com.aiprogramming.ch02;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.PriorityBlockingQueue;

/**
 * Collections for AI Data
 * 
 * This class demonstrates how to effectively use Java collections
 * for AI data structures and algorithms.
 */
public class CollectionsForAI {
    
    private static final Logger logger = LoggerFactory.getLogger(CollectionsForAI.class);
    
    public static void main(String[] args) {
        logger.info("Collections for AI Data");
        logger.info("==========================\n");
        
        // Demonstrate different collection types for AI
        demonstrateListsForSequentialData();
        demonstrateSetsForUniqueData();
        demonstrateMapsForKeyValueData();
        demonstrateQueuesForProcessing();
        demonstrateCustomDataStructures();
        demonstrateConcurrentCollections();
        
        logger.info("Collections demonstration completed!");
    }
    
    /**
     * Demonstrates using Lists for sequential data in AI
     */
    private static void demonstrateListsForSequentialData() {
        logger.info("Lists for Sequential Data:");
        
        // 1. Feature vectors
        logger.info("\n1. Feature Vectors:");
        List<Double> featureVector = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0));
        logger.info("   Feature vector: {}", featureVector);
        
        // Normalize feature vector
        double mean = featureVector.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        List<Double> normalizedVector = featureVector.stream()
            .map(x -> x - mean)
            .toList();
        logger.info("   Normalized vector: {}", normalizedVector);
        
        // 2. Time series data
        logger.info("\n2. Time Series Data:");
        List<TimeSeriesPoint> timeSeries = Arrays.asList(
            new TimeSeriesPoint(1, 10.5),
            new TimeSeriesPoint(2, 11.2),
            new TimeSeriesPoint(3, 10.8),
            new TimeSeriesPoint(4, 12.1),
            new TimeSeriesPoint(5, 11.9)
        );
        logger.info("   Time series: {}", timeSeries);
        
        // Calculate moving average
        int windowSize = 3;
        List<Double> movingAverage = calculateMovingAverage(timeSeries, windowSize);
        logger.info("   Moving average (window={}): {}", windowSize, movingAverage);
        
        // 3. Training data
        logger.info("\n3. Training Data:");
        List<DataPoint> trainingData = Arrays.asList(
            new DataPoint(new double[]{1.0, 2.0}, "class_a"),
            new DataPoint(new double[]{2.0, 3.0}, "class_a"),
            new DataPoint(new double[]{4.0, 5.0}, "class_b"),
            new DataPoint(new double[]{5.0, 6.0}, "class_b")
        );
        logger.info("   Training data size: {}", trainingData.size());
        
        // Split data into features and labels
        List<double[]> features = trainingData.stream()
            .map(DataPoint::getFeatures)
            .toList();
        List<String> labels = trainingData.stream()
            .map(DataPoint::getLabel)
            .toList();
        logger.info("   Features extracted: {} vectors", features.size());
        logger.info("   Labels extracted: {}", labels);
    }
    
    /**
     * Demonstrates using Sets for unique data in AI
     */
    private static void demonstrateSetsForUniqueData() {
        logger.info("\nSets for Unique Data:");
        
        // 1. Unique classes/labels
        logger.info("\n1. Unique Classes:");
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList("cat", "dog", "cat", "bird", "dog", "fish"));
        logger.info("   Unique classes: {}", uniqueClasses);
        logger.info("   Number of unique classes: {}", uniqueClasses.size());
        
        // 2. Feature names
        logger.info("\n2. Feature Names:");
        Set<String> featureNames = new LinkedHashSet<>(Arrays.asList(
            "age", "income", "education", "location", "age" // age appears twice
        ));
        logger.info("   Feature names (preserving order): {}", featureNames);
        
        // 3. Unique data points (for deduplication)
        logger.info("\n3. Data Deduplication:");
        List<String> rawData = Arrays.asList("A", "B", "A", "C", "B", "D", "A");
        Set<String> uniqueData = new HashSet<>(rawData);
        logger.info("   Raw data: {}", rawData);
        logger.info("   Unique data: {}", uniqueData);
        logger.info("   Removed {} duplicates", rawData.size() - uniqueData.size());
        
        // 4. Set operations for data analysis
        logger.info("\n4. Set Operations:");
        Set<String> setA = new HashSet<>(Arrays.asList("A", "B", "C", "D"));
        Set<String> setB = new HashSet<>(Arrays.asList("C", "D", "E", "F"));
        
        // Union
        Set<String> union = new HashSet<>(setA);
        union.addAll(setB);
        logger.info("   Union (A ∪ B): {}", union);
        
        // Intersection
        Set<String> intersection = new HashSet<>(setA);
        intersection.retainAll(setB);
        logger.info("   Intersection (A ∩ B): {}", intersection);
        
        // Difference
        Set<String> difference = new HashSet<>(setA);
        difference.removeAll(setB);
        logger.info("   Difference (A - B): {}", difference);
    }
    
    /**
     * Demonstrates using Maps for key-value data in AI
     */
    private static void demonstrateMapsForKeyValueData() {
        logger.info("\nMaps for Key-Value Data:");
        
        // 1. Feature importance scores
        logger.info("\n1. Feature Importance:");
        Map<String, Double> featureImportance = new HashMap<>();
        featureImportance.put("age", 0.85);
        featureImportance.put("income", 0.72);
        featureImportance.put("education", 0.63);
        featureImportance.put("location", 0.45);
        
        // Sort by importance
        List<Map.Entry<String, Double>> sortedFeatures = featureImportance.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .toList();
        logger.info("   Feature importance (sorted): {}", sortedFeatures);
        
        // 2. Word frequency counting
        logger.info("\n2. Word Frequency:");
        String text = "the quick brown fox jumps over the lazy dog the fox is quick";
        String[] words = text.toLowerCase().split("\\s+");
        
        Map<String, Integer> wordFrequency = new HashMap<>();
        for (String word : words) {
            wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
        }
        logger.info("   Word frequencies: {}", wordFrequency);
        
        // Find most frequent word
        String mostFrequent = wordFrequency.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("none");
        logger.info("   Most frequent word: '{}'", mostFrequent);
        
        // 3. Model parameters
        logger.info("\n3. Model Parameters:");
        Map<String, Object> modelParams = new LinkedHashMap<>();
        modelParams.put("learning_rate", 0.01);
        modelParams.put("batch_size", 32);
        modelParams.put("epochs", 100);
        modelParams.put("optimizer", "adam");
        modelParams.put("activation", "relu");
        
        logger.info("   Model parameters:");
        modelParams.forEach((key, value) -> logger.info("     {}: {}", key, value));
        
        // 4. Cache for computed values
        logger.info("\n4. Computation Cache:");
        Map<Integer, Long> factorialCache = new HashMap<>();
        
        // Compute factorial with caching
        int n = 10;
        long factorial = computeFactorialWithCache(n, factorialCache);
        logger.info("   Factorial of {}: {} (using cache)", n, factorial);
        logger.info("   Cache contents: {}", factorialCache);
    }
    
    /**
     * Demonstrates using Queues for processing in AI
     */
    private static void demonstrateQueuesForProcessing() {
        logger.info("\n🔄 Queues for Processing:");
        
        // 1. Batch processing queue
        logger.info("\n1. Batch Processing Queue:");
        Queue<String> batchQueue = new LinkedList<>();
        batchQueue.offer("data_batch_1");
        batchQueue.offer("data_batch_2");
        batchQueue.offer("data_batch_3");
        batchQueue.offer("data_batch_4");
        
        logger.info("   Queue size: {}", batchQueue.size());
        logger.info("   Processing batches:");
        while (!batchQueue.isEmpty()) {
            String batch = batchQueue.poll();
            logger.info("     Processing: {}", batch);
        }
        
        // 2. Priority queue for processing by importance
        logger.info("\n2. Priority Queue (by importance):");
        PriorityQueue<Task> priorityQueue = new PriorityQueue<>();
        priorityQueue.offer(new Task("high_priority_task", 1));
        priorityQueue.offer(new Task("low_priority_task", 3));
        priorityQueue.offer(new Task("medium_priority_task", 2));
        priorityQueue.offer(new Task("urgent_task", 0));
        
        logger.info("   Processing tasks by priority:");
        while (!priorityQueue.isEmpty()) {
            Task task = priorityQueue.poll();
            logger.info("     Processing: {} (priority: {})", task.getName(), task.getPriority());
        }
        
        // 3. Blocking queue for producer-consumer pattern
        logger.info("\n3. Blocking Queue (Producer-Consumer):");
        PriorityBlockingQueue<DataItem> blockingQueue = new PriorityBlockingQueue<>();
        
        // Simulate producer
        blockingQueue.offer(new DataItem("item_1", 1.0));
        blockingQueue.offer(new DataItem("item_2", 2.0));
        blockingQueue.offer(new DataItem("item_3", 0.5));
        
        logger.info("   Consuming items from blocking queue:");
        while (!blockingQueue.isEmpty()) {
            try {
                DataItem item = blockingQueue.take();
                logger.info("     Consumed: {} (value: {})", item.getName(), item.getValue());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
    
    /**
     * Demonstrates custom data structures for AI
     */
    private static void demonstrateCustomDataStructures() {
        logger.info("\n🏗️ Custom Data Structures:");
        
        // 1. Feature vector with metadata
        logger.info("\n1. Feature Vector with Metadata:");
        FeatureVector vector1 = new FeatureVector("user_1", Arrays.asList(1.0, 2.0, 3.0));
        FeatureVector vector2 = new FeatureVector("user_2", Arrays.asList(4.0, 5.0, 6.0));
        
        logger.info("   Vector 1: {}", vector1);
        logger.info("   Vector 2: {}", vector2);
        logger.info("   Similarity: {:.3f}", vector1.cosineSimilarity(vector2));
        
        // 2. Sparse vector for high-dimensional data
        logger.info("\n2. Sparse Vector:");
        SparseVector sparseVector = new SparseVector(1000);
        sparseVector.set(0, 1.0);
        sparseVector.set(100, 2.0);
        sparseVector.set(500, 3.0);
        sparseVector.set(999, 4.0);
        
        logger.info("   Sparse vector (size 1000): {}", sparseVector);
        logger.info("   Non-zero elements: {}", sparseVector.getNonZeroCount());
        logger.info("   Memory efficiency: {:.1f}%", 
            (double) sparseVector.getNonZeroCount() / 1000 * 100);
        
        // 3. Data point with multiple features
        logger.info("\n3. Multi-Feature Data Point:");
        Map<String, Object> features = new HashMap<>();
        features.put("age", 25);
        features.put("income", 50000.0);
        features.put("education", "bachelor");
        features.put("location", "urban");
        
        MultiFeatureDataPoint dataPoint = new MultiFeatureDataPoint("user_123", features, "positive");
        logger.info("   Data point: {}", dataPoint);
        logger.info("   Feature count: {}", dataPoint.getFeatureCount());
    }
    
    /**
     * Demonstrates concurrent collections for AI
     */
    private static void demonstrateConcurrentCollections() {
        logger.info("\n⚡ Concurrent Collections:");
        
        // 1. Concurrent hash map for shared data
        logger.info("\n1. Concurrent HashMap:");
        ConcurrentHashMap<String, Integer> sharedCounter = new ConcurrentHashMap<>();
        
        // Simulate concurrent updates
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            final int threadId = i;
            Thread thread = new Thread(() -> {
                for (int j = 0; j < 10; j++) {
                    sharedCounter.compute("counter", (key, value) -> (value == null) ? 1 : value + 1);
                }
                logger.info("     Thread {} completed", threadId);
            });
            threads.add(thread);
            thread.start();
        }
        
        // Wait for all threads to complete
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        logger.info("   Final counter value: {}", sharedCounter.get("counter"));
        
        // 2. Thread-safe queue for data processing
        logger.info("\n2. Thread-Safe Queue:");
        PriorityBlockingQueue<ProcessingTask> taskQueue = new PriorityBlockingQueue<>();
        
        // Add tasks
        taskQueue.offer(new ProcessingTask("task_1", 1, System.currentTimeMillis()));
        taskQueue.offer(new ProcessingTask("task_2", 2, System.currentTimeMillis()));
        taskQueue.offer(new ProcessingTask("task_3", 0, System.currentTimeMillis()));
        
        logger.info("   Tasks in queue: {}", taskQueue.size());
        logger.info("   Next task: {}", taskQueue.peek());
    }
    
    // Helper methods
    
    private static List<Double> calculateMovingAverage(List<TimeSeriesPoint> timeSeries, int windowSize) {
        List<Double> movingAverage = new ArrayList<>();
        
        for (int i = windowSize - 1; i < timeSeries.size(); i++) {
            double sum = 0.0;
            for (int j = i - windowSize + 1; j <= i; j++) {
                sum += timeSeries.get(j).getValue();
            }
            movingAverage.add(sum / windowSize);
        }
        
        return movingAverage;
    }
    
    private static long computeFactorialWithCache(int n, Map<Integer, Long> cache) {
        if (n <= 1) return 1;
        
        if (cache.containsKey(n)) {
            return cache.get(n);
        }
        
        long result = n * computeFactorialWithCache(n - 1, cache);
        cache.put(n, result);
        return result;
    }
    
    // Data classes
    
    static class TimeSeriesPoint {
        private final int timestamp;
        private final double value;
        
        public TimeSeriesPoint(int timestamp, double value) {
            this.timestamp = timestamp;
            this.value = value;
        }
        
        public int getTimestamp() { return timestamp; }
        public double getValue() { return value; }
        
        @Override
        public String toString() {
            return String.format("(%d, %.1f)", timestamp, value);
        }
    }
    
    static class DataPoint {
        private final double[] features;
        private final String label;
        
        public DataPoint(double[] features, String label) {
            this.features = features.clone();
            this.label = label;
        }
        
        public double[] getFeatures() { return features.clone(); }
        public String getLabel() { return label; }
    }
    
    static class Task implements Comparable<Task> {
        private final String name;
        private final int priority; // Lower number = higher priority
        
        public Task(String name, int priority) {
            this.name = name;
            this.priority = priority;
        }
        
        public String getName() { return name; }
        public int getPriority() { return priority; }
        
        @Override
        public int compareTo(Task other) {
            return Integer.compare(this.priority, other.priority);
        }
    }
    
    static class DataItem implements Comparable<DataItem> {
        private final String name;
        private final double value;
        
        public DataItem(String name, double value) {
            this.name = name;
            this.value = value;
        }
        
        public String getName() { return name; }
        public double getValue() { return value; }
        
        @Override
        public int compareTo(DataItem other) {
            return Double.compare(this.value, other.value);
        }
    }
    
    static class FeatureVector {
        private final String id;
        private final List<Double> features;
        
        public FeatureVector(String id, List<Double> features) {
            this.id = id;
            this.features = new ArrayList<>(features);
        }
        
        public String getId() { return id; }
        public List<Double> getFeatures() { return new ArrayList<>(features); }
        
        public double cosineSimilarity(FeatureVector other) {
            if (this.features.size() != other.features.size()) {
                throw new IllegalArgumentException("Vectors must have same dimension");
            }
            
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            
            for (int i = 0; i < features.size(); i++) {
                double a = features.get(i);
                double b = other.features.get(i);
                dotProduct += a * b;
                normA += a * a;
                normB += b * b;
            }
            
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        }
        
        @Override
        public String toString() {
            return String.format("FeatureVector{id='%s', features=%s}", id, features);
        }
    }
    
    static class SparseVector {
        private final Map<Integer, Double> nonZeroElements;
        private final int size;
        
        public SparseVector(int size) {
            this.size = size;
            this.nonZeroElements = new HashMap<>();
        }
        
        public void set(int index, double value) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException("Index: " + index);
            }
            if (value != 0.0) {
                nonZeroElements.put(index, value);
            } else {
                nonZeroElements.remove(index);
            }
        }
        
        public double get(int index) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException("Index: " + index);
            }
            return nonZeroElements.getOrDefault(index, 0.0);
        }
        
        public int getNonZeroCount() {
            return nonZeroElements.size();
        }
        
        @Override
        public String toString() {
            return String.format("SparseVector{size=%d, nonZero=%s}", size, nonZeroElements);
        }
    }
    
    static class MultiFeatureDataPoint {
        private final String id;
        private final Map<String, Object> features;
        private final String label;
        
        public MultiFeatureDataPoint(String id, Map<String, Object> features, String label) {
            this.id = id;
            this.features = new HashMap<>(features);
            this.label = label;
        }
        
        public String getId() { return id; }
        public Map<String, Object> getFeatures() { return new HashMap<>(features); }
        public String getLabel() { return label; }
        public int getFeatureCount() { return features.size(); }
        
        @Override
        public String toString() {
            return String.format("MultiFeatureDataPoint{id='%s', features=%s, label='%s'}", 
                id, features, label);
        }
    }
    
    static class ProcessingTask implements Comparable<ProcessingTask> {
        private final String name;
        private final int priority;
        private final long timestamp;
        
        public ProcessingTask(String name, int priority, long timestamp) {
            this.name = name;
            this.priority = priority;
            this.timestamp = timestamp;
        }
        
        public String getName() { return name; }
        public int getPriority() { return priority; }
        public long getTimestamp() { return timestamp; }
        
        @Override
        public int compareTo(ProcessingTask other) {
            int priorityCompare = Integer.compare(this.priority, other.priority);
            if (priorityCompare != 0) {
                return priorityCompare;
            }
            return Long.compare(this.timestamp, other.timestamp);
        }
        
        @Override
        public String toString() {
            return String.format("ProcessingTask{name='%s', priority=%d}", name, priority);
        }
    }
}

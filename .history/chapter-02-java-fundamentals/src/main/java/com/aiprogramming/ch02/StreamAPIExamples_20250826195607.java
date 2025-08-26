package com.aiprogramming.ch02;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.aiprogramming.ch02.CollectionsForAI.DataPoint;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Stream API Examples for AI Data Processing
 * 
 * This class demonstrates how to use Java Stream API for efficient
 * data processing in AI applications.
 */
public class StreamAPIExamples {
    
    private static final Logger logger = LoggerFactory.getLogger(StreamAPIExamples.class);
    
    public static void main(String[] args) {
        logger.info("Stream API Examples for AI");
        logger.info("=============================\n");
        
        // Demonstrate basic stream operations
        demonstrateBasicStreamOperations();
        
        // Show advanced stream operations
        demonstrateAdvancedStreamOperations();
        
        // Explore functional programming patterns
        demonstrateFunctionalProgrammingPatterns();
        
        // Demonstrate parallel processing
        demonstrateParallelProcessing();
        
        // Show custom collectors
        demonstrateCustomCollectors();
        
        // Demonstrate stream pipelines for AI
        demonstrateStreamPipelinesForAI();
        
        logger.info("Stream API demonstration completed!");
    }
    
    /**
     * Demonstrates basic stream operations
     */
    private static void demonstrateBasicStreamOperations() {
        logger.info("Basic Stream Operations:");
        
        // Sample dataset: sensor readings
        List<SensorReading> sensorData = Arrays.asList(
            new SensorReading("temp", 25.5, 1000),
            new SensorReading("temp", 26.1, 1001),
            new SensorReading("humidity", 60.2, 1000),
            new SensorReading("humidity", 61.0, 1001),
            new SensorReading("pressure", 1013.2, 1000),
            new SensorReading("pressure", 1012.8, 1001)
        );
        
        // 1. Filtering
        logger.info("\n1. Filtering Data:");
        List<SensorReading> temperatureReadings = sensorData.stream()
            .filter(reading -> "temp".equals(reading.getSensorType()))
            .collect(Collectors.toList());
        logger.info("   Temperature readings: {}", temperatureReadings);
        
        // 2. Mapping
        logger.info("\n2. Mapping Transformations:");
        List<Double> temperatures = sensorData.stream()
            .filter(reading -> "temp".equals(reading.getSensorType()))
            .map(SensorReading::getValue)
            .collect(Collectors.toList());
        logger.info("   Temperature values: {}", temperatures);
        
        // 3. Sorting
        logger.info("\n3. Sorting Data:");
        List<SensorReading> sortedByValue = sensorData.stream()
            .sorted(Comparator.comparing(SensorReading::getValue))
            .collect(Collectors.toList());
        logger.info("   Sorted by value: {}", sortedByValue);
        
        // 4. Limiting and skipping
        logger.info("\n4. Limiting and Skipping:");
        List<SensorReading> firstThree = sensorData.stream()
            .limit(3)
            .collect(Collectors.toList());
        logger.info("   First 3 readings: {}", firstThree);
        
        List<SensorReading> skipFirstTwo = sensorData.stream()
            .skip(2)
            .collect(Collectors.toList());
        logger.info("   Skip first 2: {}", skipFirstTwo);
        
        // 5. Finding elements
        logger.info("\n5. Finding Elements:");
        Optional<SensorReading> firstTemp = sensorData.stream()
            .filter(reading -> "temp".equals(reading.getSensorType()))
            .findFirst();
        logger.info("   First temperature reading: {}", firstTemp.orElse(null));
        
        boolean hasHighTemp = sensorData.stream()
            .filter(reading -> "temp".equals(reading.getSensorType()))
            .anyMatch(reading -> reading.getValue() > 30.0);
        logger.info("   Has high temperature (>30°C): {}", hasHighTemp);
    }
    
    /**
     * Demonstrates advanced stream operations
     */
    private static void demonstrateAdvancedStreamOperations() {
        logger.info("\n🔧 Advanced Stream Operations:");
        
        // Sample dataset: student performance
        List<Student> students = Arrays.asList(
            new Student("Alice", 85.5, "Computer Science"),
            new Student("Bob", 92.0, "Mathematics"),
            new Student("Charlie", 78.5, "Physics"),
            new Student("Diana", 95.0, "Computer Science"),
            new Student("Eve", 88.5, "Mathematics"),
            new Student("Frank", 82.0, "Physics"),
            new Student("Grace", 91.0, "Computer Science")
        );
        
        // 1. Grouping by department
        logger.info("\n1. Grouping by Department:");
        Map<String, List<Student>> studentsByDept = students.stream()
            .collect(Collectors.groupingBy(Student::getDepartment));
        logger.info("   Students by department: {}", studentsByDept);
        
        // 2. Partitioning by performance
        logger.info("\n2. Partitioning by Performance:");
        Map<Boolean, List<Student>> highPerformers = students.stream()
            .collect(Collectors.partitioningBy(student -> student.getScore() >= 90.0));
        logger.info("   High performers (≥90): {}", highPerformers.get(true));
        logger.info("   Others (<90): {}", highPerformers.get(false));
        
        // 3. Aggregating statistics
        logger.info("\n3. Aggregating Statistics:");
        DoubleSummaryStatistics stats = students.stream()
            .mapToDouble(Student::getScore)
            .summaryStatistics();
        logger.info("   Score statistics:");
        logger.info("     Count: {}", stats.getCount());
        logger.info("     Average: {:.2f}", stats.getAverage());
        logger.info("     Min: {:.2f}", stats.getMin());
        logger.info("     Max: {:.2f}", stats.getMax());
        
        // 4. Joining strings
        logger.info("\n4. Joining Strings:");
        String allNames = students.stream()
            .map(Student::getName)
            .collect(Collectors.joining(", "));
        logger.info("   All student names: {}", allNames);
        
        // 5. Reducing to a single value
        logger.info("\n5. Reducing to Single Value:");
        double totalScore = students.stream()
            .mapToDouble(Student::getScore)
            .reduce(0.0, Double::sum);
        logger.info("   Total score: {:.2f}", totalScore);
        
        Optional<Student> topStudent = students.stream()
            .reduce((s1, s2) -> s1.getScore() > s2.getScore() ? s1 : s2);
        logger.info("   Top student: {}", topStudent.orElse(null));
    }
    
    /**
     * Demonstrates functional programming patterns
     */
    private static void demonstrateFunctionalProgrammingPatterns() {
        logger.info("\n🎯 Functional Programming Patterns:");
        
        // 1. Function composition
        logger.info("\n1. Function Composition:");
        Function<Double, Double> square = x -> x * x;
        Function<Double, Double> addOne = x -> x + 1;
        Function<Double, Double> squareThenAddOne = square.andThen(addOne);
        
        List<Double> numbers = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0);
        List<Double> transformed = numbers.stream()
            .map(squareThenAddOne)
            .collect(Collectors.toList());
        logger.info("   Original: {}", numbers);
        logger.info("   Transformed (x² + 1): {}", transformed);
        
        // 2. Predicate composition
        logger.info("\n2. Predicate Composition:");
        Predicate<Integer> isEven = n -> n % 2 == 0;
        Predicate<Integer> isPositive = n -> n > 0;
        Predicate<Integer> isEvenAndPositive = isEven.and(isPositive);
        
        List<Integer> integers = Arrays.asList(-2, -1, 0, 1, 2, 3, 4, 5);
        List<Integer> filtered = integers.stream()
            .filter(isEvenAndPositive)
            .collect(Collectors.toList());
        logger.info("   Original: {}", integers);
        logger.info("   Even and positive: {}", filtered);
        
        // 3. Currying (partial application)
        logger.info("\n3. Currying (Partial Application):");
        Function<Double, Function<Double, Double>> multiply = x -> y -> x * y;
        Function<Double, Double> multiplyByTwo = multiply.apply(2.0);
        Function<Double, Double> multiplyByTen = multiply.apply(10.0);
        
        List<Double> results = numbers.stream()
            .map(multiplyByTwo)
            .collect(Collectors.toList());
        logger.info("   Original: {}", numbers);
        logger.info("   Multiplied by 2: {}", results);
        
        // 4. Higher-order functions
        logger.info("\n4. Higher-Order Functions:");
        Function<Function<Double, Double>, List<Double>> applyToAll = 
            func -> numbers.stream().map(func).collect(Collectors.toList());
        
        List<Double> squared = applyToAll.apply(square);
        List<Double> doubled = applyToAll.apply(x -> x * 2);
        logger.info("   Squared: {}", squared);
        logger.info("   Doubled: {}", doubled);
        
        // 5. Immutable data transformation
        logger.info("\n5. Immutable Data Transformation:");
        List<DataPoint> originalData = Arrays.asList(
            new DataPoint(new double[]{1.0, 2.0}, "A"),
            new DataPoint(new double[]{3.0, 4.0}, "B"),
            new DataPoint(new double[]{5.0, 6.0}, "A")
        );
        
        // Transform without modifying original
        List<DataPoint> normalizedData = originalData.stream()
            .map(dp -> normalizeDataPoint(dp))
            .collect(Collectors.toList());
        
        logger.info("   Original data: {}", originalData);
        logger.info("   Normalized data: {}", normalizedData);
    }
    
    /**
     * Demonstrates parallel processing with streams
     */
    private static void demonstrateParallelProcessing() {
        logger.info("\n⚡ Parallel Processing:");
        
        // Large dataset for demonstration
        List<Integer> largeDataset = IntStream.range(0, 1000000)
            .boxed()
            .collect(Collectors.toList());
        
        // 1. Sequential processing
        logger.info("\n1. Sequential Processing:");
        long startTime = System.currentTimeMillis();
        long sequentialSum = largeDataset.stream()
            .mapToLong(Integer::longValue)
            .sum();
        long endTime = System.currentTimeMillis();
        logger.info("   Sequential sum: {} (took {}ms)", sequentialSum, endTime - startTime);
        
        // 2. Parallel processing
        logger.info("\n2. Parallel Processing:");
        startTime = System.currentTimeMillis();
        long parallelSum = largeDataset.parallelStream()
            .mapToLong(Integer::longValue)
            .sum();
        endTime = System.currentTimeMillis();
        logger.info("   Parallel sum: {} (took {}ms)", parallelSum, endTime - startTime);
        
        // 3. Parallel processing with custom thread pool
        logger.info("\n3. Custom Thread Pool Processing:");
        // Note: In real applications, you'd use ForkJoinPool.commonPool() or custom executor
        
        // 4. Parallel processing considerations
        logger.info("\n4. Parallel Processing Considerations:");
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        
        // Stateful operations can be problematic in parallel streams
        List<String> sequentialResult = words.stream()
            .sorted()
            .collect(Collectors.toList());
        logger.info("   Sequential sorted: {}", sequentialResult);
        
        List<String> parallelResult = words.parallelStream()
            .sorted()
            .collect(Collectors.toList());
        logger.info("   Parallel sorted: {}", parallelResult);
        
        // 5. Parallel processing with reduction
        logger.info("\n5. Parallel Reduction:");
        double[] values = largeDataset.stream()
            .mapToDouble(Integer::doubleValue)
            .toArray();
        
        startTime = System.currentTimeMillis();
        double parallelAverage = Arrays.stream(values)
            .parallel()
            .average()
            .orElse(0.0);
        endTime = System.currentTimeMillis();
        logger.info("   Parallel average: {:.2f} (took {}ms)", parallelAverage, endTime - startTime);
    }
    
    /**
     * Demonstrates custom collectors
     */
    private static void demonstrateCustomCollectors() {
        logger.info("\n🔧 Custom Collectors:");
        
        List<SensorReading> readings = Arrays.asList(
            new SensorReading("temp", 25.5, 1000),
            new SensorReading("temp", 26.1, 1001),
            new SensorReading("humidity", 60.2, 1000),
            new SensorReading("pressure", 1013.2, 1000)
        );
        
        // 1. Custom collector for statistics by sensor type
        logger.info("\n1. Custom Statistics Collector:");
        Map<String, DoubleSummaryStatistics> statsByType = readings.stream()
            .collect(Collectors.groupingBy(
                SensorReading::getSensorType,
                Collectors.summarizingDouble(SensorReading::getValue)
            ));
        logger.info("   Statistics by sensor type: {}", statsByType);
        
        // 2. Custom collector for range calculation
        logger.info("\n2. Custom Range Collector:");
        Map<String, Double> rangesByType = readings.stream()
            .collect(Collectors.groupingBy(
                SensorReading::getSensorType,
                Collectors.collectingAndThen(
                    Collectors.mapping(SensorReading::getValue, Collectors.toList()),
                    values -> {
                        if (values.isEmpty()) return 0.0;
                        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
                        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
                        return max - min;
                    }
                )
            ));
        logger.info("   Ranges by sensor type: {}", rangesByType);
        
        // 3. Custom collector for weighted average
        logger.info("\n3. Custom Weighted Average Collector:");
        double weightedAverage = readings.stream()
            .collect(Collectors.collectingAndThen(
                Collectors.toList(),
                list -> {
                    if (list.isEmpty()) return 0.0;
                    double weightedSum = list.stream()
                        .mapToDouble(r -> r.getValue() * r.getTimestamp())
                        .sum();
                    double totalWeight = list.stream()
                        .mapToDouble(SensorReading::getTimestamp)
                        .sum();
                    return totalWeight > 0 ? weightedSum / totalWeight : 0.0;
                }
            ));
        logger.info("   Weighted average: {:.2f}", weightedAverage);
    }
    
    /**
     * Demonstrates stream pipelines for AI applications
     */
    private static void demonstrateStreamPipelinesForAI() {
        logger.info("\n🤖 Stream Pipelines for AI:");
        
        // Sample AI dataset: customer data
        List<Customer> customers = Arrays.asList(
            new Customer("Alice", 25, 50000, "urban", "premium"),
            new Customer("Bob", 35, 75000, "suburban", "standard"),
            new Customer("Charlie", 45, 120000, "urban", "premium"),
            new Customer("Diana", 28, 45000, "rural", "basic"),
            new Customer("Eve", 52, 95000, "suburban", "premium"),
            new Customer("Frank", 31, 60000, "urban", "standard")
        );
        
        // 1. Data preprocessing pipeline
        logger.info("\n1. Data Preprocessing Pipeline:");
        List<Customer> processedCustomers = customers.stream()
            .filter(customer -> customer.getAge() >= 18) // Remove minors
            .filter(customer -> customer.getIncome() > 0) // Remove invalid income
            .map(customer -> normalizeCustomer(customer)) // Normalize data
            .collect(Collectors.toList());
        logger.info("   Processed customers: {}", processedCustomers.size());
        
        // 2. Feature extraction pipeline
        logger.info("\n2. Feature Extraction Pipeline:");
        Map<String, Double> features = customers.stream()
            .collect(Collectors.collectingAndThen(
                Collectors.toList(),
                list -> {
                    Map<String, Double> featureMap = new HashMap<>();
                    featureMap.put("avg_age", list.stream().mapToDouble(Customer::getAge).average().orElse(0.0));
                    featureMap.put("avg_income", list.stream().mapToDouble(Customer::getIncome).average().orElse(0.0));
                    featureMap.put("premium_ratio", 
                        (double) list.stream().filter(c -> "premium".equals(c.getPlan())).count() / list.size());
                    return featureMap;
                }
            ));
        logger.info("   Extracted features: {}", features);
        
        // 3. Customer segmentation pipeline
        logger.info("\n3. Customer Segmentation Pipeline:");
        Map<String, List<Customer>> segments = customers.stream()
            .collect(Collectors.groupingBy(customer -> {
                if (customer.getIncome() > 80000) return "high_income";
                else if (customer.getIncome() > 50000) return "medium_income";
                else return "low_income";
            }));
        logger.info("   Customer segments: {}", segments.keySet());
        segments.forEach((segment, customerList) -> 
            logger.info("     {}: {} customers", segment, customerList.size()));
        
        // 4. Anomaly detection pipeline
        logger.info("\n4. Anomaly Detection Pipeline:");
        double avgIncome = customers.stream()
            .mapToDouble(Customer::getIncome)
            .average()
            .orElse(0.0);
        double stdDev = calculateStandardDeviation(
            customers.stream().mapToDouble(Customer::getIncome).toArray()
        );
        
        List<Customer> anomalies = customers.stream()
            .filter(customer -> Math.abs(customer.getIncome() - avgIncome) > 2 * stdDev)
            .collect(Collectors.toList());
        logger.info("   Anomalies (income > 2σ from mean): {}", anomalies);
        
        // 5. Recommendation pipeline
        logger.info("\n5. Recommendation Pipeline:");
        String targetLocation = "urban";
        List<Customer> recommendations = customers.stream()
            .filter(customer -> targetLocation.equals(customer.getLocation()))
            .filter(customer -> "premium".equals(customer.getPlan()))
            .sorted(Comparator.comparing(Customer::getIncome).reversed())
            .limit(3)
            .collect(Collectors.toList());
        logger.info("   Top 3 premium customers in {}: {}", targetLocation, recommendations);
    }
    
    // Helper methods
    
    private static DataPoint normalizeDataPoint(DataPoint original) {
        double[] features = original.getFeatures();
        double[] normalized = new double[features.length];
        
        // Simple min-max normalization
        double min = Arrays.stream(features).min().orElse(0.0);
        double max = Arrays.stream(features).max().orElse(1.0);
        double range = max - min;
        
        for (int i = 0; i < features.length; i++) {
            normalized[i] = range > 0 ? (features[i] - min) / range : 0.0;
        }
        
        return new DataPoint(normalized, original.getLabel());
    }
    
    private static Customer normalizeCustomer(Customer customer) {
        // Simple normalization: cap age at 100, income at 1M
        int normalizedAge = Math.min(customer.getAge(), 100);
        double normalizedIncome = Math.min(customer.getIncome(), 1000000.0);
        
        return new Customer(customer.getName(), normalizedAge, normalizedIncome, 
                          customer.getLocation(), customer.getPlan());
    }
    
    private static double calculateStandardDeviation(double[] values) {
        if (values.length == 0) return 0.0;
        
        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values)
            .map(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0.0);
        
        return Math.sqrt(variance);
    }
    
    // Data classes
    
    static class SensorReading {
        private final String sensorType;
        private final double value;
        private final long timestamp;
        
        public SensorReading(String sensorType, double value, long timestamp) {
            this.sensorType = sensorType;
            this.value = value;
            this.timestamp = timestamp;
        }
        
        public String getSensorType() { return sensorType; }
        public double getValue() { return value; }
        public long getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format("SensorReading{type='%s', value=%.1f, time=%d}", 
                sensorType, value, timestamp);
        }
    }
    
    static class Student {
        private final String name;
        private final double score;
        private final String department;
        
        public Student(String name, double score, String department) {
            this.name = name;
            this.score = score;
            this.department = department;
        }
        
        public String getName() { return name; }
        public double getScore() { return score; }
        public String getDepartment() { return department; }
        
        @Override
        public String toString() {
            return String.format("Student{name='%s', score=%.1f, dept='%s'}", 
                name, score, department);
        }
    }
    
    static class Customer {
        private final String name;
        private final int age;
        private final double income;
        private final String location;
        private final String plan;
        
        public Customer(String name, int age, double income, String location, String plan) {
            this.name = name;
            this.age = age;
            this.income = income;
            this.location = location;
            this.plan = plan;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public double getIncome() { return income; }
        public String getLocation() { return location; }
        public String getPlan() { return plan; }
        
        @Override
        public String toString() {
            return String.format("Customer{name='%s', age=%d, income=%.0f, location='%s', plan='%s'}", 
                name, age, income, location, plan);
        }
    }
}

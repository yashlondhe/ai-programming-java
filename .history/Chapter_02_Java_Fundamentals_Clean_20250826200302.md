# Chapter 2: Java Fundamentals for AI Development

Welcome to Chapter 2 of **AI Programming with Java**! This chapter focuses on essential Java concepts that are particularly important for AI and machine learning development. We'll explore how Java's features can be leveraged for efficient AI programming, from data structures to performance optimization.

## Learning Objectives

- Master Java collections framework for AI data handling
- Understand multidimensional arrays and matrix operations
- Learn Stream API for efficient data processing
- Explore functional programming concepts with lambda expressions
- Implement proper exception handling for AI applications
- Optimize memory management and performance for large datasets
- Work with Java's mathematical capabilities for AI computations

## Why Java Fundamentals Matter for AI

Java's design philosophy and features make it particularly well-suited for AI development, but understanding how to use these features effectively is crucial for building performant and maintainable AI applications.

### Java's Strengths for AI Development

**Type Safety and Compile-Time Checking:**
- Catches errors before runtime, crucial for AI applications where bugs can be expensive
- Reduces debugging time and improves code reliability
- Enables better IDE support and refactoring capabilities

**Memory Management:**
- Automatic garbage collection reduces memory-related bugs
- Predictable memory usage patterns for large datasets
- Ability to tune garbage collection for specific workloads

**Performance Characteristics:**
- Just-In-Time (JIT) compilation for optimal performance
- Efficient object creation and method dispatch
- Good performance for numerical computations

**Concurrency Support:**
- Built-in support for parallel processing
- Thread-safe collections for concurrent data access
- CompletableFuture for asynchronous programming

**Ecosystem and Libraries:**
- Rich ecosystem of AI and machine learning libraries
- Mature build tools and dependency management
- Excellent tooling for development and debugging

## Java Collections Framework for AI

The Java Collections Framework provides a comprehensive set of data structures that are essential for AI applications. Understanding when and how to use each collection type is crucial for building efficient AI systems.

### Lists for Sequential Data

Lists are fundamental for storing ordered data, which is common in AI applications such as time series, feature vectors, and training datasets.

**Use Cases in AI:**
- **Feature Vectors**: Storing numerical features for machine learning
- **Time Series Data**: Sequential data points with timestamps
- **Training Data**: Ordered collections of examples
- **Neural Network Layers**: Sequential processing of data through layers

**Key List Implementations:**

1. **ArrayList**
   - Fast random access and iteration
   - Good for feature vectors and training data
   - Memory efficient for large datasets
   - Example: Storing feature vectors for classification

2. **LinkedList**
   - Efficient insertion and deletion
   - Good for dynamic data structures
   - Useful for implementing custom algorithms
   - Example: Building custom neural network architectures

3. **Vector**
   - Thread-safe but slower than ArrayList
   - Good for concurrent access scenarios
   - Example: Shared training data in multi-threaded environments

### Sets for Unique Data

Sets are essential for handling unique elements, which is common in AI applications such as unique classes, features, or data points.

**Use Cases in AI:**
- **Unique Classes**: Storing distinct classification labels
- **Feature Names**: Maintaining unique feature identifiers
- **Data Deduplication**: Removing duplicate data points
- **Vocabulary**: Storing unique words in natural language processing

**Key Set Implementations:**

1. **HashSet**
   - Fast insertion, deletion, and lookup
   - No guaranteed order
   - Good for general-purpose unique data storage
   - Example: Storing unique class labels

2. **LinkedHashSet**
   - Maintains insertion order
   - Good for preserving data order while ensuring uniqueness
   - Example: Maintaining order of feature names

3. **TreeSet**
   - Sorted order
   - Slower than HashSet but provides ordering
   - Example: Sorted feature importance scores

### Maps for Key-Value Data

Maps are crucial for storing associations between keys and values, which is fundamental in AI applications.

**Use Cases in AI:**
- **Feature Importance**: Mapping feature names to importance scores
- **Word Frequencies**: Counting occurrences of words in text
- **Model Parameters**: Storing configuration parameters
- **Caching**: Storing computed values for reuse

**Key Map Implementations:**

1. **HashMap**
   - Fast access and modification
   - No guaranteed order
   - Good for general-purpose key-value storage
   - Example: Feature importance mapping

2. **LinkedHashMap**
   - Maintains insertion order
   - Good for preserving data order
   - Example: Ordered model parameters

3. **TreeMap**
   - Sorted by keys
   - Slower than HashMap but provides ordering
   - Example: Sorted feature importance scores

### Queues for Processing

Queues are essential for managing processing order in AI applications, particularly for batch processing and task scheduling.

**Use Cases in AI:**
- **Batch Processing**: Managing data batches for training
- **Task Scheduling**: Prioritizing AI tasks
- **Producer-Consumer Patterns**: Managing data flow
- **Event Processing**: Handling AI events in order

**Key Queue Implementations:**

1. **LinkedList (as Queue)**
   - FIFO (First-In-First-Out) behavior
   - Good for simple queuing needs
   - Example: Processing data batches in order

2. **PriorityQueue**
   - Orders elements by priority
   - Good for task scheduling
   - Example: Prioritizing AI tasks by importance

3. **ArrayBlockingQueue**
   - Thread-safe with bounded capacity
   - Good for producer-consumer patterns
   - Example: Managing data flow between threads

## Stream API and Functional Programming

The Stream API introduced in Java 8 revolutionized data processing by providing a functional approach to working with collections. This is particularly valuable for AI applications that involve extensive data manipulation.

### Functional Programming Concepts

**Immutability:**
- Data structures that cannot be modified after creation
- Reduces bugs and improves thread safety
- Enables better reasoning about code behavior

**Higher-Order Functions:**
- Functions that take other functions as parameters
- Enables composition and reuse of behavior
- Examples: map, filter, reduce operations

**Pure Functions:**
- Functions with no side effects
- Same input always produces same output
- Easier to test and reason about

### Stream API Operations

**Intermediate Operations:**
- Return new streams
- Lazy evaluation (only executed when needed)
- Can be chained together

**Terminal Operations:**
- Produce final results
- Trigger evaluation of the stream
- Cannot be chained further

### Common Stream Operations for AI

**Filtering Data:**
```java
// Filter high-performing students
List<Student> highPerformers = students.stream()
    .filter(student -> student.getScore() >= 90.0)
    .collect(Collectors.toList());
```

**Transforming Data:**
```java
// Extract feature vectors from data points
List<double[]> features = dataPoints.stream()
    .map(DataPoint::getFeatures)
    .collect(Collectors.toList());
```

**Aggregating Data:**
```java
// Calculate average score
double averageScore = students.stream()
    .mapToDouble(Student::getScore)
    .average()
    .orElse(0.0);
```

**Grouping Data:**
```java
// Group students by performance level
Map<String, List<Student>> groupedStudents = students.stream()
    .collect(Collectors.groupingBy(student -> 
        student.getScore() >= 90.0 ? "High" : 
        student.getScore() >= 70.0 ? "Medium" : "Low"));
```

### Parallel Processing with Streams

**Benefits of Parallel Streams:**
- Automatic parallelization of operations
- Improved performance for large datasets
- Simplified concurrent programming

**When to Use Parallel Streams:**
- Large datasets (typically > 10,000 elements)
- CPU-intensive operations
- Independent operations that don't share state

**Example:**
```java
// Parallel processing of large dataset
double parallelSum = largeDataset.parallelStream()
    .mapToDouble(DataPoint::getValue)
    .sum();
```

## Matrix Operations and Mathematical Computations

Matrix operations are fundamental to many AI algorithms, particularly in machine learning and deep learning. Java provides several approaches for implementing efficient matrix operations.

### Multidimensional Arrays

**2D Arrays for Matrices:**
- Simple and efficient for small to medium matrices
- Direct memory access
- Good performance for basic operations

**Example Matrix Operations:**
```java
// Matrix addition
public static double[][] addMatrices(double[][] a, double[][] b) {
    int rows = a.length;
    int cols = a[0].length;
    double[][] result = new double[rows][cols];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

// Matrix multiplication
public static double[][] multiplyMatrices(double[][] a, double[][] b) {
    int rowsA = a.length;
    int colsA = a[0].length;
    int colsB = b[0].length;
    double[][] result = new double[rowsA][colsB];
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}
```

### Vector Operations

**Dot Product:**
```java
public static double dotProduct(double[] a, double[] b) {
    if (a.length != b.length) {
        throw new IllegalArgumentException("Vectors must have same length");
    }
    
    double result = 0.0;
    for (int i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }
    return result;
}
```

**Cross Product (3D vectors):**
```java
public static double[] crossProduct(double[] a, double[] b) {
    if (a.length != 3 || b.length != 3) {
        throw new IllegalArgumentException("Cross product only defined for 3D vectors");
    }
    
    return new double[]{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}
```

### Mathematical Libraries

**Apache Commons Math:**
- Comprehensive mathematical library
- Matrix operations, statistics, optimization
- Well-documented and maintained

**ND4J (N-Dimensional Arrays for Java):**
- Fast numerical computing
- GPU acceleration support
- NumPy-like API

**EJML (Efficient Java Matrix Library):**
- High-performance matrix operations
- Optimized for scientific computing
- Good documentation and examples

## Performance Optimization for AI Applications

AI applications often deal with large datasets and computationally intensive operations. Understanding Java performance characteristics and optimization techniques is crucial for building efficient AI systems.

### Memory Management

**Object Creation and Garbage Collection:**
- Minimize object creation in hot paths
- Use object pooling for frequently created objects
- Understand garbage collection behavior

**Memory-Efficient Data Structures:**
- Use primitive arrays when possible
- Consider sparse data structures for high-dimensional data
- Implement custom data structures for specific use cases

**Example - Sparse Vector Implementation:**
```java
public class SparseVector {
    private final Map<Integer, Double> nonZeroElements;
    private final int size;
    
    public SparseVector(int size) {
        this.size = size;
        this.nonZeroElements = new HashMap<>();
    }
    
    public void set(int index, double value) {
        if (value != 0.0) {
            nonZeroElements.put(index, value);
        } else {
            nonZeroElements.remove(index);
        }
    }
    
    public double get(int index) {
        return nonZeroElements.getOrDefault(index, 0.0);
    }
}
```

### String Optimization

**String Concatenation:**
- Use StringBuilder for multiple concatenations
- Avoid string concatenation in loops
- Consider String.format for complex formatting

**Example:**
```java
// Inefficient
String result = "";
for (int i = 0; i < 10000; i++) {
    result += "item" + i + " ";
}

// Efficient
StringBuilder result = new StringBuilder();
for (int i = 0; i < 10000; i++) {
    result.append("item").append(i).append(" ");
}
```

### Boxing and Unboxing Optimization

**Primitive vs Wrapper Types:**
- Use primitive types when possible
- Avoid unnecessary boxing/unboxing
- Consider specialized collections for primitives

**Example:**
```java
// Inefficient - boxing overhead
List<Integer> boxedList = new ArrayList<>();
for (int i = 0; i < 100000; i++) {
    boxedList.add(i); // Autoboxing
}

// Efficient - primitive array
int[] primitiveArray = new int[100000];
for (int i = 0; i < 100000; i++) {
    primitiveArray[i] = i; // No boxing
}
```

### JVM Tuning for AI Workloads

**Garbage Collection Tuning:**
- Choose appropriate GC algorithm
- Tune heap size and generation sizes
- Monitor GC performance

**JVM Options for AI:**
```bash
# Increase heap size
-Xmx8g -Xms4g

# Use G1GC for large heaps
-XX:+UseG1GC

# Tune GC parameters
-XX:MaxGCPauseMillis=200
-XX:G1HeapRegionSize=16m
```

## Exception Handling for AI Applications

AI applications often deal with uncertain data and complex computations. Proper exception handling is crucial for building robust AI systems.

### Custom Exceptions for AI

**Domain-Specific Exceptions:**
- Create exceptions that reflect AI-specific errors
- Provide meaningful error messages
- Include context information

**Example:**
```java
public class InvalidFeatureVectorException extends Exception {
    public InvalidFeatureVectorException(String message) {
        super(message);
    }
    
    public InvalidFeatureVectorException(String message, Throwable cause) {
        super(message, cause);
    }
}
```

### Resource Management

**Try-with-Resources:**
- Automatic resource cleanup
- Prevents resource leaks
- Simplifies error handling

**Example:**
```java
try (DataProcessor processor = new DataProcessor()) {
    processor.processData("input.csv");
} catch (DataProcessingException e) {
    logger.error("Data processing failed", e);
}
```

### Graceful Degradation

**Error Recovery Strategies:**
- Continue processing when possible
- Provide fallback values
- Log errors for analysis

**Example:**
```java
public double safeDivision(double numerator, double denominator) {
    try {
        return numerator / denominator;
    } catch (ArithmeticException e) {
        logger.warn("Division by zero detected, returning 0.0");
        return 0.0;
    }
}
```

### Batch Processing with Error Recovery

**Robust Data Processing:**
- Process data in batches
- Handle individual failures gracefully
- Continue processing remaining data

**Example:**
```java
List<String> processedData = new ArrayList<>();
for (String dataPoint : dataPoints) {
    try {
        String processed = processDataPoint(dataPoint);
        processedData.add(processed);
    } catch (DataProcessingException e) {
        logger.warn("Skipping invalid data point: {}", dataPoint);
        // Continue processing other data points
    }
}
```

## Quick Start

1. **Navigate to Chapter 2**:
   ```bash
   cd chapter-02-java-fundamentals
   ```

2. **Install dependencies**:
   ```bash
   mvn clean install
   ```

3. **Run the main demonstration**:
   ```bash
   mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.JavaFundamentalsDemo"
   ```

## Code Examples

### Example 1: Java Fundamentals Demo
**File**: `src/main/java/com/aiprogramming/ch02/JavaFundamentalsDemo.java`

Comprehensive demonstration of Java fundamentals essential for AI development.

```java
package com.aiprogramming.ch02;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Java Fundamentals Demo for AI Development
 * 
 * This class demonstrates essential Java concepts that are particularly
 * important for AI and machine learning development.
 */
public class JavaFundamentalsDemo {
    
    private static final Logger logger = LoggerFactory.getLogger(JavaFundamentalsDemo.class);
    
    public static void main(String[] args) {
        logger.info("Java Fundamentals for AI Development");
        logger.info("=====================================\n");
        
        // Demonstrate collections for AI data
        demonstrateCollectionsForAI();
        
        // Show Stream API and functional programming
        demonstrateStreamAPI();
        
        // Explore matrix operations
        demonstrateMatrixOperations();
        
        // Performance optimization techniques
        demonstratePerformanceOptimization();
        
        // Exception handling for AI applications
        demonstrateExceptionHandling();
        
        logger.info("Java fundamentals demonstration completed!");
    }
    
    /**
     * Demonstrates Java collections framework for AI data handling
     */
    private static void demonstrateCollectionsForAI() {
        logger.info("Collections Framework for AI Data:");
        
        // 1. Lists for sequential data (e.g., time series, feature vectors)
        logger.info("\n1. Lists for Sequential Data:");
        List<Double> featureVector = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0);
        logger.info("   Feature vector: {}", featureVector);
        
        // Calculate mean using traditional approach
        double sum = 0.0;
        for (Double value : featureVector) {
            sum += value;
        }
        double mean = sum / featureVector.size();
        logger.info("   Mean (traditional): {}", String.format("%.2f", mean));
        
        // 2. Sets for unique elements (e.g., unique classes, features)
        logger.info("\n2. Sets for Unique Elements:");
        Set<String> uniqueClasses = new HashSet<>(Arrays.asList("cat", "dog", "cat", "bird", "dog"));
        logger.info("   Unique classes: {}", uniqueClasses);
        
        // 3. Maps for key-value pairs (e.g., feature importance, word frequencies)
        logger.info("\n3. Maps for Key-Value Pairs:");
        Map<String, Double> featureImportance = new HashMap<>();
        featureImportance.put("age", 0.8);
        featureImportance.put("income", 0.6);
        featureImportance.put("education", 0.4);
        logger.info("   Feature importance: {}", featureImportance);
        
        // 4. Queues for processing order (e.g., batch processing)
        logger.info("\n4. Queues for Processing Order:");
        Queue<String> processingQueue = new LinkedList<>();
        processingQueue.offer("data_point_1");
        processingQueue.offer("data_point_2");
        processingQueue.offer("data_point_3");
        logger.info("   Processing queue: {}", processingQueue);
        logger.info("   Next to process: {}", processingQueue.poll());
        
        // 5. Multidimensional arrays for matrices
        logger.info("\n5. Multidimensional Arrays for Matrices:");
        double[][] matrix = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        logger.info("   3x3 Matrix:");
        for (double[] row : matrix) {
            logger.info("   {}", Arrays.toString(row));
        }
    }
    
    /**
     * Demonstrates Stream API and functional programming concepts
     */
    private static void demonstrateStreamAPI() {
        logger.info("\nStream API and Functional Programming:");
        
        // Sample dataset: student scores
        List<Student> students = Arrays.asList(
            new Student("Alice", 85.5),
            new Student("Bob", 92.0),
            new Student("Charlie", 78.5),
            new Student("Diana", 95.0),
            new Student("Eve", 88.5)
        );
        
        // 1. Filtering data
        logger.info("\n1. Filtering Data:");
        List<Student> highPerformers = students.stream()
            .filter(student -> student.getScore() >= 90.0)
            .collect(Collectors.toList());
        logger.info("   High performers ( 90): {}", highPerformers);
        
        // 2. Mapping transformations
        logger.info("\n2. Mapping Transformations:");
        List<String> names = students.stream()
            .map(Student::getName)
            .collect(Collectors.toList());
        logger.info("   Student names: {}", names);
        
        // 3. Aggregation operations
        logger.info("\n3. Aggregation Operations:");
        double averageScore = students.stream()
            .mapToDouble(Student::getScore)
            .average()
            .orElse(0.0);
        logger.info("   Average score: {}", String.format("%.2f", averageScore));
        
        double maxScore = students.stream()
            .mapToDouble(Student::getScore)
            .max()
            .orElse(0.0);
        logger.info("   Maximum score: {}", String.format("%.2f", maxScore));
        
        // 4. Complex data processing pipeline
        logger.info("\n4. Complex Data Processing Pipeline:");
        Map<String, Double> scoreMap = students.stream()
            .filter(student -> student.getScore() >= 80.0)
            .collect(Collectors.toMap(
                Student::getName,
                Student::getScore
            ));
        logger.info("   Students with scores  80: {}", scoreMap);
        
        // 5. Parallel processing
        logger.info("\n5. Parallel Processing:");
        long startTime = System.currentTimeMillis();
        double parallelSum = students.parallelStream()
            .mapToDouble(Student::getScore)
            .sum();
        long endTime = System.currentTimeMillis();
        logger.info("   Parallel sum: {} (took {}ms)", String.format("%.2f", parallelSum), endTime - startTime);
    }
    
    /**
     * Demonstrates matrix operations essential for AI algorithms
     */
    private static void demonstrateMatrixOperations() {
        logger.info("\nMatrix Operations for AI:");
        
        // 1. Matrix creation and initialization
        logger.info("\n1. Matrix Creation:");
        int rows = 3, cols = 3;
        double[][] matrixA = new double[rows][cols];
        double[][] matrixB = new double[rows][cols];
        
        // Initialize matrices
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixA[i][j] = i + j;
                matrixB[i][j] = i * j;
            }
        }
        
        logger.info("   Matrix A:");
        printMatrix(matrixA);
        logger.info("   Matrix B:");
        printMatrix(matrixB);
        
        // 2. Matrix addition
        logger.info("\n2. Matrix Addition:");
        double[][] sumMatrix = addMatrices(matrixA, matrixB);
        logger.info("   A + B:");
        printMatrix(sumMatrix);
        
        // 3. Matrix multiplication
        logger.info("\n3. Matrix Multiplication:");
        double[][] productMatrix = multiplyMatrices(matrixA, matrixB);
        logger.info("   A   B:");
        printMatrix(productMatrix);
        
        // 4. Transpose operation
        logger.info("\n4. Matrix Transpose:");
        double[][] transposeMatrix = transposeMatrix(matrixA);
        logger.info("   A^T:");
        printMatrix(transposeMatrix);
        
        // 5. Vector operations
        logger.info("\n5. Vector Operations:");
        double[] vector1 = {1.0, 2.0, 3.0};
        double[] vector2 = {4.0, 5.0, 6.0};
        
        double dotProduct = dotProduct(vector1, vector2);
        logger.info("   Dot product: {:.2f}", dotProduct);
        
        double[] crossProduct = crossProduct(vector1, vector2);
        logger.info("   Cross product: {}", Arrays.toString(crossProduct));
    }
    
    /**
     * Demonstrates performance optimization techniques
     */
    private static void demonstratePerformanceOptimization() {
        logger.info("\nPerformance Optimization:");
        
        // 1. Memory management
        logger.info("\n1. Memory Management:");
        int largeSize = 1000000;
        
        // Inefficient approach
        long startTime = System.currentTimeMillis();
        List<Integer> inefficientList = new ArrayList<>();
        for (int i = 0; i < largeSize; i++) {
            inefficientList.add(i);
        }
        long endTime = System.currentTimeMillis();
        logger.info("   Inefficient list creation: {}ms", endTime - startTime);
        
        // Efficient approach with initial capacity
        startTime = System.currentTimeMillis();
        List<Integer> efficientList = new ArrayList<>(largeSize);
        for (int i = 0; i < largeSize; i++) {
            efficientList.add(i);
        }
        endTime = System.currentTimeMillis();
        logger.info("   Efficient list creation: {}ms", endTime - startTime);
        
        // 2. String concatenation optimization
        logger.info("\n2. String Concatenation:");
        int iterations = 10000;
        
        // Inefficient string concatenation
        startTime = System.currentTimeMillis();
        String inefficientString = "";
        for (int i = 0; i < iterations; i++) {
            inefficientString += "item" + i + " ";
        }
        endTime = System.currentTimeMillis();
        logger.info("   Inefficient concatenation: {}ms", endTime - startTime);
        
        // Efficient string concatenation with StringBuilder
        startTime = System.currentTimeMillis();
        StringBuilder efficientString = new StringBuilder();
        for (int i = 0; i < iterations; i++) {
            efficientString.append("item").append(i).append(" ");
        }
        endTime = System.currentTimeMillis();
        logger.info("   Efficient concatenation: {}ms", endTime - startTime);
        
        // 3. Boxing/unboxing optimization
        logger.info("\n3. Boxing/Unboxing Optimization:");
        
        // Inefficient boxing
        startTime = System.currentTimeMillis();
        List<Integer> boxedList = new ArrayList<>();
        for (int i = 0; i < 100000; i++) {
            boxedList.add(i); // Autoboxing
        }
        endTime = System.currentTimeMillis();
        logger.info("   Boxing overhead: {}ms", endTime - startTime);
        
        // Efficient primitive arrays
        startTime = System.currentTimeMillis();
        int[] primitiveArray = new int[100000];
        for (int i = 0; i < 100000; i++) {
            primitiveArray[i] = i; // No boxing
        }
        endTime = System.currentTimeMillis();
        logger.info("   Primitive array: {}ms", endTime - startTime);
    }
    
    /**
     * Demonstrates exception handling for AI applications
     */
    private static void demonstrateExceptionHandling() {
        logger.info("\nException Handling for AI:");
        
        // 1. Custom exceptions for AI applications
        logger.info("\n1. Custom AI Exceptions:");
        try {
            validateFeatureVector(new double[]{1.0, 2.0, -1.0, 4.0});
        } catch (InvalidFeatureVectorException e) {
            logger.error("   Feature vector validation failed: {}", e.getMessage());
        }
        
        // 2. Resource management with try-with-resources
        logger.info("\n2. Resource Management:");
        try (DataProcessor processor = new DataProcessor()) {
            processor.processData("sample_data.csv");
        } catch (DataProcessingException e) {
            logger.error("   Data processing failed: {}", e.getMessage());
        }
        
        // 3. Graceful degradation
        logger.info("\n3. Graceful Degradation:");
        double result = safeDivision(10.0, 0.0);
        logger.info("   Safe division result: {}", result);
        
        // 4. Batch processing with error recovery
        logger.info("\n4. Batch Processing with Error Recovery:");
        List<String> dataPoints = Arrays.asList("valid_data", "invalid_data", "valid_data");
        List<String> processedData = new ArrayList<>();
        
        for (String dataPoint : dataPoints) {
            try {
                String processed = processDataPoint(dataPoint);
                processedData.add(processed);
            } catch (DataProcessingException e) {
                logger.warn("   Skipping invalid data point: {}", dataPoint);
                // Continue processing other data points
            }
        }
        logger.info("   Successfully processed: {} out of {}", processedData.size(), dataPoints.size());
    }
    
    // Helper methods for matrix operations
    
    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            logger.info("   {}", Arrays.toString(row));
        }
    }
    
    private static double[][] addMatrices(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
    
    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }
    
    private static double[][] transposeMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    private static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double result = 0.0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    private static double[] crossProduct(double[] a, double[] b) {
        if (a.length != 3 || b.length != 3) {
            throw new IllegalArgumentException("Cross product only defined for 3D vectors");
        }
        
        return new double[]{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
    
    // Helper methods for exception handling
    
    private static void validateFeatureVector(double[] features) throws InvalidFeatureVectorException {
        for (int i = 0; i < features.length; i++) {
            if (features[i] < 0) {
                throw new InvalidFeatureVectorException("Negative value at index " + i + ": " + features[i]);
            }
        }
    }
    
    private static double safeDivision(double numerator, double denominator) {
        try {
            return numerator / denominator;
        } catch (ArithmeticException e) {
            logger.warn("   Division by zero detected, returning 0.0");
            return 0.0;
        }
    }
    
    private static String processDataPoint(String dataPoint) throws DataProcessingException {
        if ("invalid_data".equals(dataPoint)) {
            throw new DataProcessingException("Invalid data point: " + dataPoint);
        }
        return "processed_" + dataPoint;
    }
    
    // Inner classes and custom exceptions
    
    static class Student {
        private final String name;
        private final double score;
        
        public Student(String name, double score) {
            this.name = name;
            this.score = score;
        }
        
        public String getName() { return name; }
        public double getScore() { return score; }
        
        @Override
        public String toString() {
            return name + "(" + score + ")";
        }
    }
    
    static class InvalidFeatureVectorException extends Exception {
        public InvalidFeatureVectorException(String message) {
            super(message);
        }
    }
    
    static class DataProcessingException extends Exception {
        public DataProcessingException(String message) {
            super(message);
        }
    }
    
    static class DataProcessor implements AutoCloseable {
        public void processData(String filename) throws DataProcessingException {
            // Simulate data processing
            if (filename.contains("invalid")) {
                throw new DataProcessingException("Cannot process invalid file: " + filename);
            }
            logger.info("   Processing data from: {}", filename);
        }
        
        @Override
        public void close() {
            logger.info("   Closing data processor");
        }
    }
}
```

### Example 2: Collections for AI Data
**File**: `src/main/java/com/aiprogramming/ch02/CollectionsForAI.java`

Working with Java collections for AI data structures and algorithms.

### Example 3: Stream API and Functional Programming
**File**: `src/main/java/com/aiprogramming/ch02/StreamAPIExamples.java`

Using Stream API for efficient data processing and functional programming patterns.

### Example 4: Matrix Operations
**File**: `src/main/java/com/aiprogramming/ch02/MatrixOperations.java`

Implementing matrix operations essential for AI algorithms.

### Example 5: Performance Optimization
**File**: `src/main/java/com/aiprogramming/ch02/PerformanceOptimization.java`

Techniques for optimizing Java code for AI applications.

## Running Examples

```bash
# Main demonstration
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.JavaFundamentalsDemo"

# Collections examples
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.CollectionsForAI"

# Stream API examples
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.StreamAPIExamples"

# Matrix operations
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.MatrixOperations"

# Performance optimization
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch02.PerformanceOptimization"
```

## Advanced Java Features for AI

### Lambda Expressions and Functional Interfaces

**Functional Interfaces:**
- Single abstract method interfaces
- Enable functional programming patterns
- Built-in functional interfaces in java.util.function

**Common Functional Interfaces:**
- **Predicate<T>**: Tests a condition (boolean test(T t))
- **Function<T,R>**: Transforms data (R apply(T t))
- **Consumer<T>**: Performs an action (void accept(T t))
- **Supplier<T>**: Provides data (T get())

**Example:**
```java
// Using functional interfaces
Predicate<Double> isPositive = x -> x > 0;
Function<Double, Double> square = x -> x * x;
Consumer<String> printer = System.out::println;
Supplier<Double> randomSupplier = Math::random;
```

### Optional for Null Safety

**Benefits of Optional:**
- Explicit handling of null values
- Prevents NullPointerException
- Encourages defensive programming

**Example:**
```java
public Optional<Double> safeDivision(double numerator, double denominator) {
    if (denominator == 0) {
        return Optional.empty();
    }
    return Optional.of(numerator / denominator);
}

// Usage
Optional<Double> result = safeDivision(10.0, 2.0);
result.ifPresent(value -> System.out.println("Result: " + value));
double defaultValue = result.orElse(0.0);
```

### Concurrent Programming

**Thread Safety:**
- Use thread-safe collections when needed
- Understand synchronization mechanisms
- Consider concurrent data structures

**Example:**
```java
// Thread-safe collections
ConcurrentHashMap<String, Integer> sharedMap = new ConcurrentHashMap<>();
CopyOnWriteArrayList<String> threadSafeList = new CopyOnWriteArrayList<>();
BlockingQueue<String> blockingQueue = new ArrayBlockingQueue<>(100);
```

## Best Practices for AI Development in Java

### Code Organization

**Package Structure:**
```
com.aiprogramming.ch02/
    data/           # Data structures and models
    algorithms/     # AI algorithms
    utils/          # Utility classes
    exceptions/     # Custom exceptions
    examples/       # Example implementations
```

**Class Design:**
- Use immutable objects when possible
- Implement proper equals() and hashCode()
- Follow Java naming conventions
- Document public APIs

### Testing Strategies

**Unit Testing:**
- Test individual components
- Use mocking for dependencies
- Test edge cases and error conditions

**Integration Testing:**
- Test complete workflows
- Verify data processing pipelines
- Test performance characteristics

**Example Test:**
```java
@Test
public void testMatrixMultiplication() {
    double[][] a = {{1, 2}, {3, 4}};
    double[][] b = {{5, 6}, {7, 8}};
    double[][] expected = {{19, 22}, {43, 50}};
    
    double[][] result = MatrixOperations.multiply(a, b);
    assertArrayEquals(expected, result);
}
```

### Performance Monitoring

**Profiling Tools:**
- JProfiler, YourKit, VisualVM
- Built-in JVM monitoring
- Application Performance Monitoring (APM)

**Key Metrics:**
- Memory usage and garbage collection
- CPU utilization
- Response times
- Throughput

## Exercises

### Exercise 1: Custom Data Structure (Beginner)
Create a custom data structure for storing feature vectors with methods for:
- Adding features
- Computing similarity between vectors
- Finding nearest neighbors

**Requirements:**
- Implement a FeatureVector class with similarity calculations
- Support different similarity metrics (cosine, euclidean, manhattan)
- Implement efficient nearest neighbor search
- Include comprehensive unit tests

**Detailed Instructions:**
1. **FeatureVector Class**: Create a class to represent feature vectors
   - Store features as a list of doubles
   - Implement similarity calculations
   - Support vector operations (addition, subtraction, scaling)

2. **Similarity Metrics**: Implement multiple similarity measures
   - Cosine similarity
   - Euclidean distance
   - Manhattan distance
   - Pearson correlation

3. **Nearest Neighbor Search**: Implement efficient search algorithms
   - Brute force search
   - K-d tree for higher dimensions
   - Locality-sensitive hashing for approximate search

4. **Testing**: Create comprehensive tests
   - Unit tests for each method
   - Performance benchmarks
   - Edge case testing

### Exercise 2: Data Pipeline with Streams (Intermediate)
Build a data processing pipeline using Stream API that:
- Loads data from CSV files
- Filters and transforms the data
- Performs basic statistical calculations
- Outputs processed results

**Requirements:**
- Implement CSV data loading
- Create data transformation pipelines
- Calculate statistical measures
- Handle errors gracefully

**Detailed Instructions:**
1. **CSV Loading**: Implement CSV file reading
   - Parse CSV format
   - Handle different data types
   - Validate data integrity
   - Support large files efficiently

2. **Data Transformation**: Create transformation pipelines
   - Filter invalid data
   - Transform data types
   - Normalize numerical data
   - Handle missing values

3. **Statistical Calculations**: Implement statistical functions
   - Mean, median, mode
   - Standard deviation, variance
   - Correlation analysis
   - Distribution analysis

4. **Error Handling**: Implement robust error handling
   - Handle malformed data
   - Log errors appropriately
   - Provide meaningful error messages
   - Continue processing when possible

### Exercise 3: Matrix Library (Advanced)
Implement a basic matrix library with:
- Matrix multiplication
- Eigenvalue decomposition
- Singular value decomposition (SVD)
- Performance benchmarking

**Requirements:**
- Implement core matrix operations
- Use efficient algorithms
- Support different matrix types
- Include performance optimizations

**Detailed Instructions:**
1. **Core Operations**: Implement basic matrix operations
   - Addition, subtraction, multiplication
   - Transpose, inverse
   - Determinant calculation
   - Rank computation

2. **Advanced Operations**: Implement complex operations
   - Eigenvalue decomposition
   - Singular value decomposition
   - QR decomposition
   - Cholesky decomposition

3. **Performance Optimization**: Optimize for performance
   - Use efficient algorithms
   - Implement parallel processing
   - Optimize memory usage
   - Cache intermediate results

4. **Benchmarking**: Create performance benchmarks
   - Compare with existing libraries
   - Measure memory usage
   - Profile performance bottlenecks
   - Document performance characteristics

## Chapter Summary

**Key Concepts Learned:**
- Java collections framework for AI data handling
- Stream API and functional programming concepts
- Matrix operations and mathematical computations
- Performance optimization techniques
- Memory management for large datasets
- Exception handling for AI applications

**Technical Skills:**
- Efficient data structure usage for AI applications
- Functional programming patterns with Java
- Mathematical computations and matrix operations
- Performance profiling and optimization
- Robust exception handling for AI systems

**Code Examples:**
- Comprehensive Java fundamentals demonstration
- Collections framework for AI data structures
- Stream API for functional data processing
- Matrix operations implementation
- Performance optimization techniques

**Practical Knowledge:**
- Understanding of Java's strengths for AI development
- Best practices for AI application development
- Performance considerations for large-scale AI systems
- Error handling strategies for robust AI applications

## Additional Resources

### Official Documentation
- [Java Collections Framework](https://docs.oracle.com/javase/tutorial/collections/)
- [Java Stream API](https://docs.oracle.com/javase/8/docs/api/java/util/stream/package-summary.html)
- [Apache Commons Math](https://commons.apache.org/proper/commons-math/)
- [Google Guava](https://github.com/google/guava)

### Books and References
- "Effective Java" by Joshua Bloch
- "Java Concurrency in Practice" by Brian Goetz
- "Functional Programming in Java" by Venkat Subramaniam
- "Java Performance" by Scott Oaks

### Online Resources
- [Baeldung Java Tutorials](https://www.baeldung.com/)
- [Java Code Geeks](https://www.javacodegeeks.com/)
- [DZone Java](https://dzone.com/java-jdk-development-tutorials-tools-news)
- [Stack Overflow Java](https://stackoverflow.com/questions/tagged/java)

### Tools and Libraries
- [JProfiler](https://www.ej-technologies.com/products/jprofiler/overview.html)
- [YourKit](https://www.yourkit.com/)
- [VisualVM](https://visualvm.github.io/)
- [Apache Commons Collections](https://commons.apache.org/proper/commons-collections/)

## Next Steps

In the next chapter, we'll explore machine learning fundamentals, including:
- **Machine Learning Workflow**: Understanding the complete ML process
- **Data Preprocessing**: Cleaning and preparing data for ML
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Evaluation**: Assessing model performance
- **Overfitting and Underfitting**: Understanding model generalization

This foundation in Java fundamentals will enable you to implement machine learning algorithms efficiently and build robust AI applications.

---

**Next Chapter**: Chapter 3: Machine Learning Basics

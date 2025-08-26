# Chapter 6: Unsupervised Learning

This chapter implements various unsupervised learning algorithms in Java, demonstrating how to discover patterns in data without labeled examples.

## 🎯 Learning Objectives

- Understand unsupervised learning concepts and applications
- Implement clustering algorithms (K-Means, DBSCAN)
- Learn about dimensionality reduction techniques (PCA)
- Explore association rule learning (Apriori algorithm)
- Master evaluation metrics for unsupervised learning
- Understand when to use different algorithms

## 🚀 Quick Start

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Building and Running

```bash
# Compile the project
mvn clean compile

# Run the main demo
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch06.UnsupervisedLearningDemo"

# Run tests
mvn test

# Create executable JAR
mvn clean package
```

## 📚 Algorithms Implemented

### 1. Clustering Algorithms

#### K-Means Clustering
- **File**: `KMeans.java`
- **Purpose**: Partition data into k clusters based on similarity
- **Features**:
  - K-means++ initialization for better convergence
  - Elbow method for optimal k selection
  - Configurable max iterations and tolerance
  - Silhouette score and inertia calculation

```java
// Example usage
KMeans kmeans = new KMeans(3);
ClusteringResult result = kmeans.fit(dataPoints);
result.printStatistics();

// Find optimal k
int optimalK = KMeans.findOptimalK(dataPoints, 10);
```

#### DBSCAN Clustering
- **File**: `DBSCAN.java`
- **Purpose**: Density-based clustering for arbitrary-shaped clusters
- **Features**:
  - Automatic epsilon parameter estimation
  - Noise point identification
  - Density-reachable cluster expansion
  - Works with non-spherical clusters

```java
// Example usage
double epsilon = DBSCAN.findOptimalEpsilon(dataPoints, 5);
DBSCAN dbscan = new DBSCAN(epsilon, 5);
ClusteringResult result = dbscan.fit(dataPoints);
```

### 2. Dimensionality Reduction

#### Principal Component Analysis (PCA)
- **File**: `PCA.java`
- **Purpose**: Reduce data dimensionality while preserving variance
- **Features**:
  - Linear dimensionality reduction
  - Explained variance calculation
  - Forward and inverse transformations
  - Data centering and scaling

```java
// Example usage
PCA pca = new PCA(2);
List<DataPoint> reducedData = pca.fitTransform(highDimData);
List<DataPoint> reconstructed = pca.inverseTransform(reducedData);
```

### 3. Association Rule Learning

#### Apriori Algorithm
- **File**: `Apriori.java`
- **Purpose**: Discover frequent itemsets and association rules
- **Features**:
  - Frequent itemset mining
  - Association rule generation
  - Support and confidence thresholds
  - Market basket analysis

```java
// Example usage
Apriori apriori = new Apriori(0.1, 0.5); // 10% support, 50% confidence
apriori.fit(transactions);
List<Set<String>> frequentItemsets = apriori.getFrequentItemsets();
List<Apriori.AssociationRule> rules = apriori.getAssociationRules();
```

## 📊 Data Structures

### DataPoint
- **File**: `DataPoint.java`
- **Purpose**: Represents a single data point with features
- **Features**:
  - Multi-dimensional feature vector
  - Distance calculations (Euclidean, Manhattan, Cosine)
  - Cluster assignment tracking
  - Unique identifier

### ClusteringResult
- **File**: `ClusteringResult.java`
- **Purpose**: Holds clustering results and evaluation metrics
- **Features**:
  - Cluster assignments and centroids
  - Silhouette score calculation
  - Inertia (within-cluster sum of squares)
  - Statistical summaries

## 🧪 Demo Output

The main demo showcases:

1. **Clustering Comparison**:
   - K-Means vs DBSCAN performance
   - Optimal parameter selection
   - Evaluation metrics comparison

2. **Dimensionality Reduction**:
   - High-dimensional data compression
   - Reconstruction quality assessment
   - Explained variance analysis

3. **Association Rule Mining**:
   - Frequent itemset discovery
   - Rule generation and evaluation
   - Market basket analysis

## 📈 Performance Metrics

### Clustering Evaluation
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1)
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Cluster Sizes**: Distribution of points across clusters

### Dimensionality Reduction Evaluation
- **Reconstruction Error**: Quality of data reconstruction
- **Explained Variance**: Proportion of variance captured
- **Compression Ratio**: Original vs reduced dimensions

### Association Rule Evaluation
- **Support**: Frequency of itemset occurrence
- **Confidence**: Reliability of association rules
- **Lift**: Independence measure of rule strength

## 🔧 Configuration

### Algorithm Parameters

#### K-Means
- `k`: Number of clusters
- `maxIterations`: Maximum training iterations (default: 100)
- `tolerance`: Convergence threshold (default: 1e-4)

#### DBSCAN
- `epsilon`: Neighborhood radius
- `minPoints`: Minimum points for core cluster (default: 5)

#### PCA
- `nComponents`: Number of principal components
- Must be ≤ original feature dimensions

#### Apriori
- `minSupport`: Minimum support threshold (0-1)
- `minConfidence`: Minimum confidence threshold (0-1)

## 🎯 Use Cases

### Clustering Applications
- **Customer Segmentation**: Group customers by behavior
- **Image Segmentation**: Separate image regions
- **Document Clustering**: Organize text documents
- **Anomaly Detection**: Identify outliers

### Dimensionality Reduction Applications
- **Data Visualization**: 2D/3D plotting of high-dimensional data
- **Feature Engineering**: Create new features
- **Data Compression**: Reduce storage requirements
- **Noise Reduction**: Remove irrelevant dimensions

### Association Rule Applications
- **Market Basket Analysis**: Product recommendations
- **Cross-selling**: Suggest related items
- **Inventory Management**: Optimize product placement
- **Web Usage Mining**: Page navigation patterns

## 🚀 Advanced Features

### Parameter Optimization
- **Elbow Method**: Automatic k selection for K-Means
- **Epsilon Estimation**: Optimal DBSCAN parameter selection
- **Grid Search**: Systematic parameter tuning

### Evaluation Framework
- **Multiple Metrics**: Comprehensive algorithm comparison
- **Statistical Analysis**: Confidence intervals and significance
- **Visualization Support**: Ready for plotting libraries

### Extensibility
- **Modular Design**: Easy to add new algorithms
- **Interface Consistency**: Standardized API across algorithms
- **Configuration Management**: Flexible parameter handling

## 📝 Code Examples

### Customer Segmentation
```java
// Load customer data
List<DataPoint> customers = loadCustomerData();

// Find optimal number of segments
int optimalSegments = KMeans.findOptimalK(customers, 10);

// Perform clustering
KMeans kmeans = new KMeans(optimalSegments);
ClusteringResult segments = kmeans.fit(customers);

// Analyze segments
segments.printStatistics();
```

### Product Recommendation
```java
// Load transaction data
List<Set<String>> transactions = loadTransactionData();

// Mine association rules
Apriori apriori = new Apriori(0.05, 0.3);
apriori.fit(transactions);

// Get recommendations
List<Apriori.AssociationRule> rules = apriori.getAssociationRules();
for (Apriori.AssociationRule rule : rules) {
    System.out.println(rule);
}
```

### Data Compression
```java
// Load high-dimensional data
List<DataPoint> highDimData = loadHighDimensionalData();

// Compress to 2D for visualization
PCA pca = new PCA(2);
List<DataPoint> compressedData = pca.fitTransform(highDimData);

// Calculate compression quality
double reconstructionError = calculateReconstructionError(highDimData, 
    pca.inverseTransform(compressedData));
```

## 🔍 Troubleshooting

### Common Issues
1. **Empty Clusters**: Reduce k or adjust data preprocessing
2. **Poor Clustering**: Check data scaling and feature relevance
3. **High Reconstruction Error**: Increase number of components
4. **No Association Rules**: Lower support/confidence thresholds

### Performance Tips
- **Data Preprocessing**: Normalize features for better clustering
- **Parameter Tuning**: Use cross-validation for optimal settings
- **Memory Management**: Process large datasets in batches
- **Algorithm Selection**: Choose based on data characteristics

## 📚 Further Reading

### Theory
- **Clustering**: K-means, DBSCAN, Hierarchical clustering
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Association Rules**: Apriori, FP-Growth, Eclat

### Applications
- **Business Intelligence**: Customer analytics, market research
- **Computer Vision**: Image segmentation, feature extraction
- **Natural Language Processing**: Document clustering, topic modeling
- **Bioinformatics**: Gene expression analysis, protein clustering

## 🤝 Contributing

### Adding New Algorithms
1. Implement the algorithm class
2. Add comprehensive tests
3. Update the demo with examples
4. Document parameters and usage

### Testing
```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=KMeansTest

# Generate test coverage
mvn jacoco:report
```

## 📄 License

This project is part of the AI Programming with Java book. See the main project license for details.

---

**Next Steps**: 
- Explore Chapter 7 (Neural Networks) for deep learning
- Experiment with different datasets and parameters
- Implement additional unsupervised learning algorithms
- Build real-world applications using these algorithms

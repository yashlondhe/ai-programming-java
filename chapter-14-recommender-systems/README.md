# Chapter 14: Recommender Systems

This chapter covers the implementation of various recommender system algorithms in Java, including collaborative filtering, content-based filtering, and hybrid approaches.

## Overview

Recommender systems are algorithms designed to suggest relevant items to users based on their preferences and behavior. This chapter implements several key approaches:

- **Collaborative Filtering**: User-based and item-based approaches
- **Content-Based Filtering**: Based on item features and user preferences
- **Matrix Factorization**: SVD and other dimensionality reduction techniques
- **Hybrid Systems**: Combining multiple approaches for better recommendations

## Project Structure

```
src/main/java/com/aiprogramming/ch14/
├── core/
│   ├── RecommenderSystem.java          # Base interface for recommender systems
│   ├── Rating.java                     # Rating data structure
│   ├── User.java                       # User representation
│   ├── Item.java                       # Item representation
│   └── RecommendationResult.java       # Recommendation result wrapper
├── collaborative/
│   ├── UserBasedCollaborativeFiltering.java
│   ├── ItemBasedCollaborativeFiltering.java
│   └── MatrixFactorization.java
├── content/
│   ├── ContentBasedFiltering.java
│   └── FeatureVector.java
├── hybrid/
│   ├── HybridRecommender.java
│   └── WeightedHybridRecommender.java
├── evaluation/
│   ├── RecommenderEvaluator.java
│   ├── PrecisionRecall.java
│   └── MeanAbsoluteError.java
├── data/
│   ├── DataLoader.java
│   └── MovieLensDataLoader.java
└── examples/
    ├── MovieRecommenderExample.java
    ├── ProductRecommenderExample.java
    └── MusicRecommenderExample.java
```

## Key Components

### Core Classes

- **RecommenderSystem**: Base interface that all recommender implementations must implement
- **Rating**: Represents a user's rating for an item
- **User**: User representation with preferences and ratings
- **Item**: Item representation with features and metadata

### Collaborative Filtering

- **UserBasedCollaborativeFiltering**: Finds similar users and recommends items they liked
- **ItemBasedCollaborativeFiltering**: Finds similar items and recommends them
- **MatrixFactorization**: Uses SVD and other techniques to learn latent factors

### Content-Based Filtering

- **ContentBasedFiltering**: Recommends items similar to what the user has liked before
- **FeatureVector**: Represents item features for similarity calculations

### Hybrid Systems

- **HybridRecommender**: Combines multiple recommender approaches
- **WeightedHybridRecommender**: Uses weighted combinations of different approaches

### Evaluation

- **RecommenderEvaluator**: Framework for evaluating recommender performance
- **PrecisionRecall**: Calculates precision and recall metrics
- **MeanAbsoluteError**: Calculates MAE for rating predictions

## Getting Started

### Prerequisites

- Java 11 or higher
- Maven 3.6 or higher

### Building the Project

```bash
mvn clean compile
```

### Running Examples

```bash
# Run movie recommender example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch14.MovieRecommenderExample"

# Run product recommender example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch14.ProductRecommenderExample"

# Run music recommender example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch14.MusicRecommenderExample"
```

### Running Tests

```bash
mvn test
```

## Usage Examples

### Basic Movie Recommendation

```java
// Load data
MovieLensDataLoader loader = new MovieLensDataLoader();
List<Rating> ratings = loader.loadRatings("data/ratings.csv");

// Create recommender
UserBasedCollaborativeFiltering recommender = new UserBasedCollaborativeFiltering(ratings);

// Get recommendations for user 1
List<RecommendationResult> recommendations = recommender.recommend(1, 10);
```

### Content-Based Filtering

```java
// Create content-based recommender
ContentBasedFiltering recommender = new ContentBasedFiltering();

// Add item features
recommender.addItem(1, new FeatureVector("action", "adventure", "sci-fi"));
recommender.addItem(2, new FeatureVector("comedy", "romance"));

// Get recommendations
List<RecommendationResult> recommendations = recommender.recommend(userPreferences, 5);
```

### Hybrid Recommendation

```java
// Create hybrid recommender
HybridRecommender hybrid = new HybridRecommender();
hybrid.addRecommender(new UserBasedCollaborativeFiltering(ratings), 0.6);
hybrid.addRecommender(new ContentBasedFiltering(), 0.4);

// Get hybrid recommendations
List<RecommendationResult> recommendations = hybrid.recommend(userId, 10);
```

## Evaluation Metrics

The project includes several evaluation metrics:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual ratings
- **Root Mean Square Error (RMSE)**: Square root of average squared differences

## Data Formats

### Rating Data
CSV format with columns: userId, itemId, rating, timestamp

### Item Features
CSV format with columns: itemId, feature1, feature2, ..., featureN

### User Preferences
CSV format with columns: userId, preference1, preference2, ..., preferenceN

## Performance Considerations

- **Memory Usage**: Large datasets may require significant memory
- **Computation Time**: Collaborative filtering can be computationally expensive
- **Scalability**: Consider using approximate nearest neighbor algorithms for large datasets

## Extending the Framework

To add new recommender algorithms:

1. Implement the `RecommenderSystem` interface
2. Override the `recommend` method
3. Add appropriate evaluation metrics
4. Create example usage in the examples package

## Contributing

When contributing to this chapter:

1. Follow the existing code style and patterns
2. Add comprehensive unit tests
3. Update documentation and examples
4. Ensure all tests pass before submitting

## License

This project is part of the AI Programming with Java book and follows the same licensing terms.

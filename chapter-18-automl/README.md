# Chapter 18: AutoML and Neural Architecture Search

This chapter implements Automated Machine Learning (AutoML) and Neural Architecture Search (NAS) capabilities in Java. The framework provides automated hyperparameter tuning, feature selection, model selection, and neural network architecture optimization.

## Overview

The AutoML framework consists of several key components:

1. **AutoML Pipeline**: Complete automated machine learning workflow
2. **Hyperparameter Optimization**: Bayesian optimization, grid search, and random search
3. **Feature Selection**: Multiple feature selection algorithms
4. **Model Selection**: Automated model comparison and selection
5. **Neural Architecture Search**: Automated neural network architecture discovery

## Project Structure

```
src/main/java/com/aiprogramming/ch18/
├── AutoML.java                           # Main AutoML pipeline
├── AutoMLConfig.java                     # AutoML configuration
├── AutoMLDemo.java                       # Comprehensive demo
├── AutoMLResult.java                     # AutoML results
├── DataSplit.java                        # Data splitting utilities
├── FeatureSelectionConfig.java           # Feature selection config
├── FeatureSelectionResult.java           # Feature selection results
├── FeatureSelector.java                  # Feature selection algorithms
├── HyperparameterOptimizationConfig.java # Hyperparameter optimization config
├── HyperparameterOptimizationResult.java # Hyperparameter optimization results
├── HyperparameterOptimizer.java          # Hyperparameter optimization algorithms
├── HyperparameterTrial.java              # Individual trial results
├── LinearRegression.java                 # Linear regression model
├── LogisticRegression.java               # Logistic regression model
├── MLModel.java                          # Model interface
├── ModelSelectionConfig.java             # Model selection config
├── ModelSelectionResult.java             # Model selection results
├── ModelSelector.java                    # Model selection algorithms
├── NASConfig.java                        # Neural Architecture Search config
├── NeuralArchitectureSearch.java         # Neural Architecture Search
├── NeuralNetwork.java                    # Neural network model
└── RandomForest.java                     # Random forest model
```

## Features

### AutoML Pipeline
- **Complete Workflow**: Feature selection → Model selection → Hyperparameter optimization
- **Configurable**: Customizable parameters for each step
- **Cross-validation**: Built-in cross-validation support
- **Time Budgeting**: Configurable time limits for optimization

### Hyperparameter Optimization
- **Bayesian Optimization**: Efficient optimization using acquisition functions
- **Grid Search**: Systematic parameter space exploration
- **Random Search**: Random parameter sampling
- **Multiple Algorithms**: Support for various optimization strategies

### Feature Selection
- **Correlation-based**: Select features based on correlation with target
- **Mutual Information**: Information-theoretic feature selection
- **Recursive Feature Elimination**: Iterative feature removal
- **Lasso Regularization**: Sparse feature selection
- **Random Forest Importance**: Tree-based feature importance

### Model Selection
- **Multiple Models**: Linear regression, logistic regression, random forest, neural networks
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, MSE, MAE, R²
- **Cross-validation**: Robust model evaluation
- **Automated Comparison**: Automatic best model selection

### Neural Architecture Search
- **Architecture Generation**: Random neural network architectures
- **Layer Configuration**: Configurable layer sizes, activations, dropout
- **Search Space**: Customizable architecture search space
- **Performance Evaluation**: Automated architecture evaluation

## Compilation and Running

### Prerequisites
- Java 11 or higher
- Maven (for dependency management)

### Compilation
```bash
# Navigate to the chapter directory
cd chapter-18-automl

# Compile the project
javac -cp "src/main/java" src/main/java/com/aiprogramming/ch18/*.java
```

### Running the Demo
```bash
# Run the main demo
java -cp "src/main/java" com.aiprogramming.ch18.AutoMLDemo
```

## Usage Examples

### Basic AutoML Pipeline
```java
// Configure AutoML
AutoMLConfig config = AutoMLConfig.builder()
    .maxTrials(50)
    .cvFolds(5)
    .build();

// Create AutoML instance
AutoML autoML = new AutoML(config);

// Run complete pipeline
AutoMLResult result = autoML.runAutoML(features, targets);

// Print results
result.printSummary();
```

### Hyperparameter Optimization
```java
// Configure optimization
HyperparameterOptimizationConfig config = new HyperparameterOptimizationConfig(
    HyperparameterOptimizationConfig.OptimizationMethod.BAYESIAN, 
    20, 5, 42L, 1e-6
);

// Create optimizer
HyperparameterOptimizer optimizer = new HyperparameterOptimizer(config);

// Optimize model
LinearRegression model = new LinearRegression();
HyperparameterOptimizationResult result = optimizer.optimize(model, features, targets);

// Get best parameters
Map<String, Object> bestParams = result.getBestParameters();
```

### Feature Selection
```java
// Configure feature selection
FeatureSelectionConfig config = new FeatureSelectionConfig(
    FeatureSelectionConfig.SelectionMethod.CORRELATION, 
    10, 0.01, true
);

// Create selector
FeatureSelector selector = new FeatureSelector(config);

// Select features
FeatureSelectionResult result = selector.selectFeatures(features, targets);

// Get selected features
double[][] selectedFeatures = result.getSelectedFeatures();
int[] selectedIndices = result.getSelectedFeatureIndices();
```

### Neural Architecture Search
```java
// Configure NAS
NASConfig config = new NASConfig(30, 1, 5, 10, 100, 20, 1, 0.5,
                                new String[]{"relu", "tanh"}, 42L);

// Create NAS instance
NeuralArchitectureSearch nas = new NeuralArchitectureSearch(config);

// Search for optimal architecture
NeuralArchitectureSearch.NASResult result = nas.search(features, targets);

// Get best architecture
NeuralArchitectureSearch.NeuralArchitecture bestArch = result.getBestArchitecture();
```

### Model Selection
```java
// Configure model selection
ModelSelectionConfig config = new ModelSelectionConfig();
ModelSelector selector = new ModelSelector(config);

// Select best model
ModelSelectionResult result = selector.selectBestModel(features, targets);

// Get best model
MLModel bestModel = result.getBestModel();
```

## Configuration Options

### AutoML Configuration
- `maxTrials`: Maximum number of optimization trials
- `maxTimeSeconds`: Maximum time budget for optimization
- `cvFolds`: Number of cross-validation folds

### Hyperparameter Optimization
- `optimizationMethod`: BAYESIAN, GRID_SEARCH, or RANDOM_SEARCH
- `maxTrials`: Maximum optimization trials
- `cvFolds`: Cross-validation folds
- `randomSeed`: Random seed for reproducibility

### Feature Selection
- `selectionMethod`: CORRELATION, MUTUAL_INFORMATION, RECURSIVE_FEATURE_ELIMINATION, LASSO, RANDOM_FOREST_IMPORTANCE
- `maxFeatures`: Maximum number of features to select
- `threshold`: Feature importance threshold
- `useCrossValidation`: Whether to use cross-validation

### Model Selection
- `candidateModels`: Set of model classes to evaluate
- `evaluationMetric`: Metric for model comparison
- `useCrossValidation`: Whether to use cross-validation
- `cvFolds`: Number of cross-validation folds

### Neural Architecture Search
- `maxTrials`: Maximum architecture trials
- `minLayers`/`maxLayers`: Layer count range
- `minLayerSize`/`maxLayerSize`: Layer size range
- `activationFunctions`: Available activation functions
- `maxDropout`: Maximum dropout rate

## Performance Considerations

### Optimization Strategies
- **Random Search**: Fastest, good for initial exploration
- **Grid Search**: Systematic but computationally expensive
- **Bayesian Optimization**: Most efficient for expensive evaluations

### Time Budgeting
- Set appropriate `maxTrials` based on available time
- Use `maxTimeSeconds` for time-constrained scenarios
- Consider early stopping for long-running optimizations

### Scalability
- Feature selection reduces dimensionality for faster training
- Cross-validation provides robust performance estimates
- Parallel evaluation can be implemented for faster optimization

## Best Practices

### Data Preparation
- Normalize/standardize features before AutoML
- Handle missing values appropriately
- Split data into train/validation/test sets

### Configuration
- Start with reasonable defaults
- Adjust parameters based on dataset size and complexity
- Use domain knowledge to guide feature selection

### Evaluation
- Use multiple evaluation metrics
- Validate results on holdout test set
- Consider model interpretability requirements

### Production Deployment
- Save best models and configurations
- Implement model versioning
- Monitor model performance over time

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `maxTrials` or use smaller datasets
2. **Slow Performance**: Use faster optimization methods or reduce search space
3. **Poor Results**: Check data quality and feature engineering
4. **Compilation Errors**: Ensure Java 11+ and correct classpath

### Debugging
- Enable verbose output in configuration
- Check intermediate results during optimization
- Validate data preprocessing steps

## Extending the Framework

### Adding New Models
1. Implement the `MLModel` interface
2. Add model class to `ModelSelectionConfig`
3. Implement hyperparameter support

### Adding New Feature Selection Methods
1. Add method to `FeatureSelectionConfig.SelectionMethod`
2. Implement selection logic in `FeatureSelector`
3. Update configuration handling

### Adding New Optimization Algorithms
1. Add method to `HyperparameterOptimizationConfig.OptimizationMethod`
2. Implement optimization logic in `HyperparameterOptimizer`
3. Update parameter generation methods

## References

- AutoML: Hutter, F., Kotthoff, L., & Vanschoren, J. (2019). Automated Machine Learning
- Neural Architecture Search: Elsken, T., et al. (2019). Neural Architecture Search: A Survey
- Hyperparameter Optimization: Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization

## License

This code is part of the AI Programming with Java book project and follows the same licensing terms.

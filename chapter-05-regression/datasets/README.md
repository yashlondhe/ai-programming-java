# Chapter 5: Regression Datasets

This directory contains sample datasets for testing and demonstrating regression algorithms.

## 📊 Available Datasets

### 1. House Prices Dataset
- **File**: `house_prices_sample.csv`
- **Description**: Synthetic house price data with multiple features
- **Features**: square_feet, bedrooms, bathrooms, age_years, lot_size_acres, garage_spaces
- **Target**: price_usd
- **Samples**: 1000 records
- **Use Case**: Demonstrating multiple linear regression, polynomial regression, and regularization

### 2. Boston Housing Dataset
- **File**: `boston_housing.csv`
- **Description**: Classic Boston housing dataset (simplified version)
- **Features**: Various housing characteristics
- **Target**: median_home_value
- **Use Case**: Comparative analysis with well-known dataset

### 3. Student Performance Dataset
- **File**: `student_performance.csv`
- **Description**: Student exam scores based on study hours and other factors
- **Features**: study_hours, attendance_rate, previous_score, sleep_hours
- **Target**: exam_score
- **Use Case**: Simple regression examples and education applications

### 4. Stock Price Dataset
- **File**: `stock_price_features.csv`
- **Description**: Stock price prediction features
- **Features**: volume, moving_avg_10, moving_avg_30, volatility, rsi
- **Target**: next_day_close
- **Use Case**: Time series and financial regression examples

## 🔧 Data Format

All datasets follow a consistent CSV format:
- First row contains column headers
- Numeric values only (pre-processed)
- Missing values handled or removed
- Features normalized where appropriate

## 📝 Usage Examples

### Loading Data in Java

```java
// Using built-in CSV reader
List<RegressionDataPoint> data = loadDataFromCSV("datasets/house_prices_sample.csv");

// Manual loading
public static List<RegressionDataPoint> loadDataFromCSV(String filename) {
    List<RegressionDataPoint> dataPoints = new ArrayList<>();
    
    try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
        String[] headers = br.readLine().split(",");
        String targetColumn = headers[headers.length - 1]; // Last column is target
        
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            Map<String, Double> features = new HashMap<>();
            
            // Add all features except the last column (target)
            for (int i = 0; i < values.length - 1; i++) {
                features.put(headers[i], Double.parseDouble(values[i]));
            }
            
            double target = Double.parseDouble(values[values.length - 1]);
            dataPoints.add(new RegressionDataPoint(features, target));
        }
    } catch (IOException e) {
        throw new RuntimeException("Error loading data: " + e.getMessage());
    }
    
    return dataPoints;
}
```

### Quick Data Analysis

```java
// Basic statistics
RegressionDataAnalyzer analyzer = new RegressionDataAnalyzer();
DataStatistics stats = analyzer.analyzeData(data);
stats.printSummary();

// Feature correlations
Map<String, Double> correlations = analyzer.calculateTargetCorrelations(data);
correlations.entrySet().stream()
    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
    .forEach(entry -> 
        System.out.printf("%s: %.3f%n", entry.getKey(), entry.getValue()));
```

## 🏗️ Data Generation

The synthetic datasets were generated using the following principles:

### House Prices
- Base price formula with realistic coefficients
- Added non-linear relationships (diminishing returns, depreciation)
- Gaussian noise for realism
- Correlated features (e.g., bedrooms vs. square feet)

### Student Performance
- Linear relationship with study hours as primary factor
- Interaction effects between features
- Bounded target values (0-100 score range)

### Stock Price Features
- Technical indicators based on historical patterns
- Volatility and momentum features
- Realistic noise levels for financial data

## 📊 Data Quality

All datasets have been validated for:
- ✅ No missing values
- ✅ Reasonable feature ranges
- ✅ Target variable distribution
- ✅ Feature-target correlations
- ✅ No duplicate records
- ✅ Consistent formatting

## 🔄 Updating Datasets

To regenerate or update datasets:

```bash
# Regenerate all datasets
gulp generate:datasets

# Generate specific dataset
gulp generate:houses
gulp generate:students
gulp generate:stocks
```

## 📈 Expected Results

When using these datasets with our regression algorithms, you should expect:

| Dataset | Best Algorithm | Expected R² | Key Features |
|---------|---------------|-------------|--------------|
| House Prices | Ridge/Lasso | 0.85-0.95 | square_feet, bedrooms |
| Student Performance | Linear | 0.75-0.85 | study_hours, attendance |
| Stock Price | SVR/Polynomial | 0.60-0.75 | moving_avg, volatility |

## 🔍 Data Exploration

Use the provided data exploration utilities:

```java
// Feature importance analysis
FeatureAnalyzer analyzer = new FeatureAnalyzer();
analyzer.analyzeFeatureImportance(data);

// Outlier detection
OutlierDetector detector = new OutlierDetector();
List<Integer> outliers = detector.detectOutliers(data);

// Distribution analysis
DistributionAnalyzer distAnalyzer = new DistributionAnalyzer();
distAnalyzer.plotFeatureDistributions(data);
```

## 📚 References

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)
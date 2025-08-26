# Datasets for AI Programming with Java

This directory contains sample datasets used throughout the book examples and exercises.

## 📁 Dataset Organization

```
datasets/
├── classification/
│   ├── iris.csv              # Iris flower dataset
│   ├── spam.csv              # Email spam dataset
│   └── credit_card.csv       # Credit card fraud dataset
├── regression/
│   ├── housing.csv           # Boston housing dataset
│   ├── stock_prices.csv      # Stock price data
│   └── weather.csv           # Weather forecasting data
├── clustering/
│   ├── customers.csv         # Customer segmentation data
│   └── countries.csv         # Country statistics
├── nlp/
│   ├── reviews.csv           # Product reviews
│   ├── news_articles.csv     # News articles
│   └── tweets.csv            # Twitter sentiment data
├── time_series/
│   ├── sales.csv             # Sales data
│   ├── temperature.csv       # Temperature data
│   └── energy.csv            # Energy consumption data
└── images/
    ├── digits/               # Handwritten digits
    ├── faces/                # Face images
    └── objects/              # Object images
```

## 📊 Dataset Descriptions

### Classification Datasets

#### Iris Dataset (`classification/iris.csv`)
- **Source**: UCI Machine Learning Repository
- **Description**: 150 samples of iris flowers with 4 features
- **Features**: Sepal length, sepal width, petal length, petal width
- **Target**: Species (setosa, versicolor, virginica)
- **Use Case**: Chapter 4 - Classification algorithms

#### Spam Dataset (`classification/spam.csv`)
- **Source**: UCI Machine Learning Repository
- **Description**: Email spam detection dataset
- **Features**: Word frequencies, character counts
- **Target**: Spam (1) or Ham (0)
- **Use Case**: Chapter 4 - Text classification

#### Credit Card Fraud (`classification/credit_card.csv`)
- **Source**: Kaggle
- **Description**: Credit card transaction data
- **Features**: Transaction amount, time, location
- **Target**: Fraudulent (1) or Legitimate (0)
- **Use Case**: Chapter 4 - Anomaly detection

### Regression Datasets

#### Housing Dataset (`regression/housing.csv`)
- **Source**: UCI Machine Learning Repository
- **Description**: Boston housing prices
- **Features**: Crime rate, room size, age, etc.
- **Target**: House price
- **Use Case**: Chapter 5 - Regression algorithms

#### Stock Prices (`regression/stock_prices.csv`)
- **Source**: Yahoo Finance
- **Description**: Historical stock price data
- **Features**: Open, high, low, close, volume
- **Target**: Future price prediction
- **Use Case**: Chapter 5 - Time series regression

### Clustering Datasets

#### Customer Segmentation (`clustering/customers.csv`)
- **Source**: Synthetic data
- **Description**: Customer purchase behavior
- **Features**: Age, income, spending score
- **Use Case**: Chapter 6 - Customer segmentation

### NLP Datasets

#### Product Reviews (`nlp/reviews.csv`)
- **Source**: Amazon product reviews
- **Description**: Customer product reviews
- **Features**: Review text, rating, product category
- **Target**: Sentiment (positive/negative)
- **Use Case**: Chapter 10 - Sentiment analysis

#### News Articles (`nlp/news_articles.csv`)
- **Source**: News API
- **Description**: News articles from various sources
- **Features**: Article text, title, category
- **Use Case**: Chapter 10 - Text classification

### Time Series Datasets

#### Sales Data (`time_series/sales.csv`)
- **Source**: Synthetic data
- **Description**: Monthly sales data
- **Features**: Date, sales amount, product category
- **Use Case**: Chapter 15 - Time series forecasting

## 🔧 Using Datasets

### Loading Datasets in Java

```java
// Example: Loading CSV dataset
public class DatasetLoader {
    
    public static List<DataPoint> loadCSV(String filePath) {
        List<DataPoint> data = new ArrayList<>();
        
        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            String[] header = reader.readNext(); // Skip header
            
            String[] line;
            while ((line = reader.readNext()) != null) {
                DataPoint point = parseDataPoint(line);
                data.add(point);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error loading dataset: " + filePath, e);
        }
        
        return data;
    }
    
    private static DataPoint parseDataPoint(String[] values) {
        // Implementation depends on dataset structure
        return new DataPoint(values);
    }
}
```

### Dataset Preprocessing

```java
// Example: Data preprocessing utilities
public class DataPreprocessor {
    
    public static double[][] normalize(double[][] data) {
        // Min-max normalization
        double[][] normalized = new double[data.length][data[0].length];
        
        for (int j = 0; j < data[0].length; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            
            // Find min and max for each feature
            for (double[] row : data) {
                min = Math.min(min, row[j]);
                max = Math.max(max, row[j]);
            }
            
            // Normalize
            for (int i = 0; i < data.length; i++) {
                normalized[i][j] = (data[i][j] - min) / (max - min);
            }
        }
        
        return normalized;
    }
    
    public static double[][] standardize(double[][] data) {
        // Z-score standardization
        double[][] standardized = new double[data.length][data[0].length];
        
        for (int j = 0; j < data[0].length; j++) {
            double sum = 0;
            double sumSquared = 0;
            
            // Calculate mean
            for (double[] row : data) {
                sum += row[j];
            }
            double mean = sum / data.length;
            
            // Calculate standard deviation
            for (double[] row : data) {
                sumSquared += Math.pow(row[j] - mean, 2);
            }
            double std = Math.sqrt(sumSquared / data.length);
            
            // Standardize
            for (int i = 0; i < data.length; i++) {
                standardized[i][j] = (data[i][j] - mean) / std;
            }
        }
        
        return standardized;
    }
}
```

## 📋 Dataset Requirements

### File Formats
- **CSV**: Comma-separated values for tabular data
- **JSON**: For structured data and metadata
- **Images**: PNG/JPG for computer vision examples
- **Text**: TXT files for NLP examples

### Data Quality
- **Clean**: Remove or handle missing values
- **Consistent**: Standardize formats and units
- **Balanced**: Ensure fair representation across classes
- **Anonymized**: Remove sensitive information

### Size Guidelines
- **Small**: < 1MB for quick examples
- **Medium**: 1-10MB for standard exercises
- **Large**: > 10MB for performance testing (optional)

## 🔒 Privacy and Ethics

### Data Privacy
- All datasets are anonymized
- No personally identifiable information (PII)
- Synthetic data used where appropriate
- Public datasets with proper attribution

### Ethical Considerations
- Fair representation across demographics
- Bias detection and mitigation
- Responsible AI development practices
- Transparency in data sources

## 📚 Additional Resources

### Dataset Sources
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [AWS Open Data Registry](https://registry.opendata.aws/)

### Data Processing Tools
- [Apache Commons CSV](https://commons.apache.org/proper/commons-csv/)
- [Jackson](https://github.com/FasterXML/jackson) for JSON
- [OpenCSV](http://opencsv.sourceforge.net/)
- [Apache POI](https://poi.apache.org/) for Excel files

---

**Note**: Some datasets may be large and not included in the repository. In such cases, download instructions and preprocessing scripts are provided in the respective chapter directories.

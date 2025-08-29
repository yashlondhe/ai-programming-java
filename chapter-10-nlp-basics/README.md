# Chapter 10: Text Processing and Analysis

This chapter implements fundamental Natural Language Processing (NLP) techniques in Java, covering text preprocessing, document vectorization, word embeddings, and basic NLP applications.

## Overview

This project provides a complete implementation of NLP basics with the following features:

- **Text Preprocessing**: Tokenization, stemming, stop word removal, and text cleaning
- **Document Vectorization**: Bag of Words and TF-IDF representations
- **Word Embeddings**: Simplified Word2Vec implementation for word vectors
- **Sentiment Analysis**: Naive Bayes-based sentiment classifier
- **Named Entity Recognition**: Pattern-based entity extraction
- **Text Similarity**: Cosine similarity and document matching
- **NLP Applications**: Complete examples for each technique

## Project Structure

```
src/main/java/com/aiprogramming/ch10/
├── TextPreprocessor.java              # Text preprocessing utilities
├── BagOfWords.java                    # Bag of Words document representation
├── TFIDF.java                         # TF-IDF document vectorization
├── Word2Vec.java                      # Word embeddings implementation
├── SentimentAnalyzer.java             # Sentiment analysis classifier
├── NamedEntityRecognizer.java         # Named entity recognition
├── TextPreprocessingExample.java      # Text preprocessing demonstration
├── TFIDFExample.java                  # TF-IDF usage examples
├── SentimentAnalysisExample.java      # Sentiment analysis examples
├── NERExample.java                    # Named entity recognition examples
└── Word2VecExample.java               # Word2Vec examples
```

## Key Components

### 1. TextPreprocessor
Handles basic text preprocessing tasks:
- **Text Cleaning**: Remove punctuation, convert to lowercase, normalize whitespace
- **Tokenization**: Split text into individual words/tokens
- **Stop Word Removal**: Remove common words that don't carry meaning
- **Stemming**: Reduce words to their root form (simplified Porter algorithm)
- **N-gram Generation**: Create word sequences of specified length
- **Sentence Extraction**: Split text into sentences

### 2. BagOfWords
Simple document vectorization:
- **Vocabulary Building**: Create word-to-index mapping from documents
- **Frequency Counting**: Count word occurrences in documents
- **Vector Generation**: Convert documents to numerical vectors
- **Feature Names**: Map vector dimensions back to words

### 3. TFIDF
Advanced document vectorization:
- **Term Frequency**: Normalize word counts by document length
- **Inverse Document Frequency**: Weight words by their rarity across documents
- **Document Similarity**: Compute cosine similarity between documents
- **Feature Importance**: Identify most distinctive words

### 4. Word2Vec
Word embedding implementation:
- **Skip-gram Training**: Learn word vectors from context
- **Vector Similarity**: Find semantically similar words
- **Vocabulary Management**: Handle large vocabularies efficiently
- **Training Visualization**: Monitor training progress

### 5. SentimentAnalyzer
Sentiment classification:
- **Naive Bayes**: Probabilistic classification approach
- **TF-IDF Features**: Use TF-IDF vectors as features
- **Confidence Scoring**: Provide prediction confidence
- **Model Evaluation**: Accuracy, precision, recall, F1-score

### 6. NamedEntityRecognizer
Entity extraction:
- **Pattern Matching**: Regex-based entity detection
- **Entity Types**: Person, Organization, Location, Date, Money, Percent, Time
- **Position Tracking**: Extract entities with their text positions
- **Entity Validation**: Heuristic-based entity verification

## Usage Examples

### Text Preprocessing
```java
TextPreprocessor preprocessor = new TextPreprocessor();

// Clean and tokenize text
String text = "The quick brown fox jumps over the lazy dog!";
List<String> tokens = preprocessor.preprocess(text, true, true);

// Remove stop words and apply stemming
System.out.println(tokens); // [quick, brown, fox, jump, lazi, dog]
```

### TF-IDF Document Vectorization
```java
TFIDF tfidf = new TFIDF();

// Sample documents
List<String> documents = Arrays.asList(
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing helps computers understand human language.",
    "Deep learning uses neural networks with multiple layers."
);

// Fit and transform
List<double[]> vectors = tfidf.fitTransform(documents, true, false);

// Find similar documents
String query = "machine learning algorithms";
double[] queryVector = tfidf.transformDocument(query, true, false);
List<Integer> similarDocs = tfidf.findMostSimilarDocuments(queryVector, vectors, 2);
```

### Sentiment Analysis
```java
SentimentAnalyzer analyzer = new SentimentAnalyzer();

// Training data
List<String> positiveTexts = Arrays.asList("Great product!", "Amazing service!");
List<String> negativeTexts = Arrays.asList("Terrible quality", "Poor experience");

// Train model
analyzer.train(positiveTexts, negativeTexts);

// Predict sentiment
String text = "This product is really good!";
double sentiment = analyzer.predictSentiment(text); // Returns 0.0 to 1.0
String sentimentClass = analyzer.predictSentimentClass(text); // "positive" or "negative"
```

### Named Entity Recognition
```java
NamedEntityRecognizer ner = new NamedEntityRecognizer();

String text = "John Smith works at Microsoft Corporation in Seattle, WA.";
Map<String, List<String>> entities = ner.extractEntities(text);

// Extract entities with positions
List<NamedEntityRecognizer.Entity> entitiesWithPositions = 
    ner.extractEntitiesWithPositions(text);
```

### Word2Vec Word Embeddings
```java
Word2Vec word2vec = new Word2Vec(50, 2, 0.01); // 50-dim vectors, window=2, lr=0.01

// Training documents
List<String> documents = Arrays.asList(
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing helps computers understand human language."
);

// Build vocabulary and train
word2vec.buildVocabulary(documents, 1);
word2vec.train(documents, 50);

// Find similar words
List<String> similarWords = word2vec.findMostSimilarWords("machine", 5);
```

## Running the Examples

### Compile and Run
```bash
# Compile the project
mvn compile

# Run text preprocessing example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch10.TextPreprocessingExample"

# Run TF-IDF example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch10.TFIDFExample"

# Run sentiment analysis example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch10.SentimentAnalysisExample"

# Run NER example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch10.NERExample"

# Run Word2Vec example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch10.Word2VecExample"
```

### Individual Examples

#### Text Preprocessing Example
Demonstrates:
- Text cleaning and normalization
- Tokenization and stop word removal
- Stemming and n-gram generation
- Vocabulary building from multiple documents

#### TF-IDF Example
Shows:
- Document vectorization with TF-IDF
- Document similarity computation
- Feature importance analysis
- Comparison with Bag of Words

#### Sentiment Analysis Example
Illustrates:
- Training a sentiment classifier
- Predicting sentiment scores and classes
- Model evaluation metrics
- Confidence-based classification

#### NER Example
Demonstrates:
- Entity extraction from text
- Pattern matching for different entity types
- Position tracking for entities
- Entity statistics and analysis

#### Word2Vec Example
Shows:
- Training word embeddings
- Finding semantically similar words
- Vector analysis and visualization
- Semantic relationship exploration

## Key Features

### Text Preprocessing
- **Comprehensive Cleaning**: Remove punctuation, numbers, extra whitespace
- **Flexible Tokenization**: Support for different tokenization strategies
- **Stop Word Lists**: Configurable stop word removal
- **Stemming Algorithms**: Simplified Porter stemming implementation
- **N-gram Support**: Generate word sequences of any length

### Document Vectorization
- **Bag of Words**: Simple frequency-based representation
- **TF-IDF**: Advanced weighting scheme for better document representation
- **Similarity Metrics**: Cosine similarity for document comparison
- **Feature Analysis**: Identify important words and features

### Word Embeddings
- **Skip-gram Model**: Learn word vectors from context
- **Configurable Parameters**: Vector size, window size, learning rate
- **Similarity Search**: Find semantically similar words
- **Training Monitoring**: Track training progress and loss

### Sentiment Analysis
- **Naive Bayes Classifier**: Probabilistic approach to sentiment classification
- **TF-IDF Features**: Use document vectors as features
- **Confidence Scoring**: Provide prediction confidence levels
- **Model Evaluation**: Comprehensive evaluation metrics

### Named Entity Recognition
- **Multiple Entity Types**: Person, Organization, Location, Date, Money, Percent, Time
- **Pattern Matching**: Regex-based entity detection
- **Position Tracking**: Extract entities with their text positions
- **Entity Validation**: Heuristic-based entity verification

## Dependencies

- **Apache Commons Math**: Mathematical operations and utilities
- **Apache Commons Lang**: Text processing utilities
- **Apache Commons Collections**: Data structure utilities
- **OpenNLP**: Natural language processing tools
- **Stanford CoreNLP**: Advanced NLP capabilities
- **Jackson**: JSON processing
- **OpenCSV**: CSV file handling
- **JUnit**: Unit testing

## Performance Considerations

### Text Preprocessing
- **Efficient Tokenization**: Optimized string splitting and processing
- **Memory Management**: Stream-based processing for large documents
- **Caching**: Cache compiled regex patterns for better performance

### Document Vectorization
- **Sparse Representations**: Efficient storage for high-dimensional vectors
- **Incremental Processing**: Support for large document collections
- **Memory Optimization**: Minimize memory usage for large vocabularies

### Word Embeddings
- **Training Efficiency**: Optimized training loops and gradient computation
- **Vocabulary Management**: Efficient handling of large vocabularies
- **Vector Operations**: Optimized similarity computations

## Extensibility

### Adding New Preprocessing Steps
```java
// Extend TextPreprocessor with custom preprocessing
public class CustomPreprocessor extends TextPreprocessor {
    public List<String> customPreprocess(String text) {
        // Add custom preprocessing logic
        return super.preprocess(text, true, true);
    }
}
```

### Custom Entity Types
```java
// Add new entity patterns to NamedEntityRecognizer
ner.addEntityPattern("CUSTOM_TYPE", Arrays.asList("pattern1", "pattern2"));
```

### Custom Similarity Metrics
```java
// Implement custom similarity functions
public double customSimilarity(double[] vector1, double[] vector2) {
    // Custom similarity computation
    return similarityScore;
}
```

## Best Practices

### Text Preprocessing
1. **Consistent Cleaning**: Apply the same preprocessing to training and test data
2. **Stop Word Selection**: Choose stop words appropriate for your domain
3. **Stemming Strategy**: Consider whether stemming improves your specific task
4. **N-gram Selection**: Choose appropriate n-gram sizes for your application

### Document Vectorization
1. **Vocabulary Size**: Balance vocabulary size with computational efficiency
2. **Feature Selection**: Consider removing low-frequency terms
3. **Normalization**: Apply appropriate normalization for your use case
4. **Similarity Metrics**: Choose similarity metrics appropriate for your data

### Word Embeddings
1. **Training Data**: Use sufficient and relevant training data
2. **Hyperparameters**: Tune vector size, window size, and learning rate
3. **Evaluation**: Evaluate embeddings on relevant downstream tasks
4. **Domain Adaptation**: Consider domain-specific training when possible

### Sentiment Analysis
1. **Training Data Quality**: Use high-quality, balanced training data
2. **Feature Engineering**: Consider domain-specific features
3. **Evaluation Metrics**: Use appropriate metrics for your application
4. **Confidence Thresholds**: Set appropriate confidence thresholds for predictions

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce vocabulary size
   - Use sparse vector representations
   - Process documents in batches

2. **Poor Sentiment Analysis Performance**
   - Check training data quality and balance
   - Increase training data size
   - Tune preprocessing parameters

3. **Slow Word2Vec Training**
   - Reduce vector size or window size
   - Use smaller vocabulary
   - Reduce number of training epochs

4. **Entity Recognition Issues**
   - Review and update entity patterns
   - Add domain-specific patterns
   - Validate entity extraction results

### Performance Optimization

1. **Use Efficient Data Structures**
   - HashMap for vocabulary lookups
   - ArrayList for dynamic collections
   - Arrays for fixed-size vectors

2. **Optimize String Operations**
   - Use StringBuilder for string concatenation
   - Compile regex patterns once
   - Minimize string allocations

3. **Memory Management**
   - Clear unused references
   - Use appropriate collection sizes
   - Consider streaming for large datasets

## Future Enhancements

### Planned Features
- **Advanced Stemming**: Implement full Porter or Snowball stemming
- **Part-of-Speech Tagging**: Add POS tagging capabilities
- **Dependency Parsing**: Implement dependency parsing
- **Advanced NER**: Add machine learning-based NER
- **Word Sense Disambiguation**: Implement WSD algorithms
- **Topic Modeling**: Add LDA and other topic modeling algorithms

### Integration Opportunities
- **Deep Learning**: Integrate with neural network frameworks
- **Cloud Services**: Add support for cloud-based NLP services
- **Real-time Processing**: Implement streaming NLP capabilities
- **Multi-language Support**: Add support for multiple languages

## Conclusion

This chapter provides a solid foundation for Natural Language Processing in Java. The implemented components cover the essential NLP techniques needed for text analysis, document processing, and language understanding. The modular design allows for easy extension and customization for specific use cases.

The examples demonstrate practical applications of each technique, making it easy to understand and apply these concepts to real-world problems. The comprehensive documentation and best practices guide help ensure successful implementation and deployment of NLP solutions.

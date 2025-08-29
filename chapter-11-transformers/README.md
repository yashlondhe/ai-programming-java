# Chapter 11: Advanced NLP with Transformers

This chapter implements transformer architecture and BERT models from scratch in Java, covering the fundamental concepts of attention mechanisms, transformer layers, and modern NLP techniques.

## Overview

This project provides a complete implementation of transformer models with the following features:

- **Attention Mechanisms**: Scaled dot-product attention and multi-head attention
- **Transformer Architecture**: Complete encoder-decoder transformer with positional encoding
- **BERT Model**: Bidirectional encoder with masked language modeling and next sentence prediction
- **Tokenizer**: Simple tokenizer for text preprocessing
- **Text Generation**: Autoregressive text generation capabilities
- **Question Answering**: BERT-based question answering framework

## Project Structure

```
src/main/java/com/aiprogramming/ch11/
├── Attention.java                    # Attention mechanism implementations
├── TransformerLayer.java             # Encoder/decoder layers and feed-forward networks
├── Transformer.java                  # Complete transformer model
├── BERT.java                         # BERT model implementation
├── Tokenizer.java                    # Text tokenization utilities
└── TransformerExample.java           # Main example demonstrating all features
```

## Key Components

### 1. Attention Mechanisms
- **ScaledDotProductAttention**: Core attention computation with scaling and masking
- **MultiHeadAttention**: Multi-head attention with parallel attention heads
- **Attention Masks**: Causal and padding masks for different use cases

### 2. Transformer Layers
- **EncoderLayer**: Self-attention + feed-forward network with residual connections
- **DecoderLayer**: Self-attention + cross-attention + feed-forward network
- **FeedForwardNetwork**: Two-layer feed-forward network with ReLU activation
- **LayerNormalization**: Layer normalization for training stability
- **PositionalEncoding**: Sinusoidal positional encoding for sequence position information

### 3. Transformer Model
- **Complete Architecture**: Full encoder-decoder transformer
- **Token Embeddings**: Learnable token embeddings
- **Text Generation**: Autoregressive generation with greedy decoding
- **Parameter Management**: Model parameter access and modification

### 4. BERT Model
- **Bidirectional Encoder**: Encoder-only architecture for bidirectional understanding
- **Masked Language Modeling**: Predict masked tokens in context
- **Next Sentence Prediction**: Binary classification for sentence relationships
- **Special Tokens**: CLS, SEP, PAD, MASK, UNK tokens
- **Segment Embeddings**: Different embeddings for sentence pairs

### 5. Tokenizer
- **Text Tokenization**: Word-level and subword tokenization
- **BERT Encoding**: Special token handling for BERT
- **Sentence Pair Encoding**: Support for question-answer and sentence pair tasks
- **Attention Masks**: Automatic attention mask generation

## Usage Examples

### Basic Transformer
```java
// Create transformer model
int dModel = 64;
int numLayers = 2;
int numHeads = 4;
int dff = 128;
int vocabSize = 1000;
int maxSeqLen = 50;

Transformer transformer = new Transformer(dModel, numLayers, numHeads, dff, vocabSize, maxSeqLen);

// Create input data
RealMatrix encoderInput = new Array2DRowRealMatrix(1, 10);
RealMatrix decoderInput = new Array2DRowRealMatrix(1, 8);
// ... fill with token IDs

// Create masks
RealMatrix encoderMask = new Array2DRowRealMatrix(10, 10);
RealMatrix decoderMask = Attention.MultiHeadAttention.createCausalMask(8);

// Forward pass
RealMatrix output = transformer.forward(encoderInput, decoderInput, encoderMask, decoderMask);
```

### BERT Model
```java
// Create BERT model
BERT bert = new BERT(128, 2, 4, 256, 1000, 64);

// Create tokenizer
Tokenizer tokenizer = new Tokenizer(1000, 64);

// Encode text
String sentenceA = "The cat sat on the mat.";
String sentenceB = "It was a sunny day.";

List<Integer> inputIds = tokenizer.encodeSentencePair(sentenceA, sentenceB);
List<Integer> segmentIds = tokenizer.createSegmentIds(sentenceA, sentenceB);
List<Integer> attentionMask = tokenizer.createAttentionMask(inputIds);

// Convert to matrices and run BERT
RealMatrix inputMatrix = convertToMatrix(inputIds);
RealMatrix segmentMatrix = convertToMatrix(segmentIds);
RealMatrix maskMatrix = createAttentionMaskMatrix(attentionMask);

BERT.BERTOutput output = bert.forward(inputMatrix, segmentMatrix, maskMatrix);
```

### Text Generation
```java
// Generate text with transformer
RealMatrix encoderInput = createContextMatrix();
RealMatrix encoderMask = new Array2DRowRealMatrix(5, 5);

int startToken = 101;
int endToken = 102;
int maxLength = 10;

List<Integer> generated = transformer.generate(encoderInput, encoderMask, maxLength, startToken, endToken);
```

### Masked Language Modeling
```java
// Create MLM training data
RealMatrix inputIds = createInputMatrix();
double maskProbability = 0.15;

BERT.MLMTrainingData mlmData = BERT.createMLMData(inputIds, maskProbability);

// Predict masked tokens
RealMatrix predictions = bert.predictMaskedTokens(
    mlmData.maskedIds, segmentIds, attentionMask, mlmData.maskedPositions);
```

## Building and Running

### Prerequisites
- Java 11 or higher
- Maven 3.6 or higher

### Build the Project
```bash
mvn clean compile
```

### Run Examples
```bash
# Run main transformer example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch11.TransformerExample"
```

### Run Tests
```bash
mvn test
```

## Key Concepts Covered

1. **Attention Mechanisms**: How transformers attend to different parts of the input
2. **Multi-Head Attention**: Parallel attention heads for different types of relationships
3. **Positional Encoding**: How transformers handle sequence order information
4. **Residual Connections**: Skip connections for gradient flow
5. **Layer Normalization**: Normalization for training stability
6. **Masked Language Modeling**: BERT's pretraining objective
7. **Next Sentence Prediction**: BERT's sentence relationship learning
8. **Text Generation**: Autoregressive generation with transformers
9. **Question Answering**: Using BERT for extractive QA tasks

## Learning Objectives

By working with this code, you will understand:

- How attention mechanisms work and why they're powerful
- The complete transformer architecture (encoder-decoder)
- How BERT differs from standard transformers
- The importance of positional encoding in transformers
- How masked language modeling enables bidirectional understanding
- How to implement text generation with transformers
- The role of different attention masks (causal, padding)
- How to tokenize text for transformer models
- How to fine-tune pre-trained models for specific tasks

## Exercises

The chapter includes several exercises to reinforce learning:

1. **Attention Visualization**: Implement attention weight visualization
2. **Custom Tokenizer**: Build a more sophisticated tokenizer
3. **Fine-tuning BERT**: Implement fine-tuning for classification tasks
4. **Text Summarization**: Build a transformer-based summarizer
5. **Machine Translation**: Implement encoder-decoder for translation
6. **Named Entity Recognition**: Use BERT for NER tasks
7. **Sentiment Analysis**: Fine-tune BERT for sentiment classification
8. **Model Compression**: Implement knowledge distillation for smaller models

## Advanced Topics

### 1. Model Optimization
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision**: Faster training with reduced precision
- **Model Parallelism**: Distributed training across multiple devices

### 2. Advanced Architectures
- **GPT Models**: Decoder-only transformers for generation
- **T5 Models**: Text-to-text transfer transformers
- **RoBERTa**: Optimized BERT pretraining
- **ALBERT**: Lightweight BERT with parameter sharing

### 3. Training Techniques
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Regularization**: Dropout and weight decay
- **Data Augmentation**: Text augmentation techniques
- **Curriculum Learning**: Progressive difficulty training

## Real-world Applications

1. **Machine Translation**: Google Translate, DeepL
2. **Text Generation**: GPT models, ChatGPT
3. **Question Answering**: SQuAD, conversational AI
4. **Text Classification**: Sentiment analysis, topic classification
5. **Named Entity Recognition**: Information extraction
6. **Text Summarization**: News summarization, document summarization
7. **Code Generation**: GitHub Copilot, code completion
8. **Conversational AI**: Chatbots, virtual assistants

## Performance Considerations

- **Memory Usage**: Transformers require significant memory for large models
- **Computational Complexity**: Attention scales quadratically with sequence length
- **Training Time**: Large models require extensive training time
- **Inference Speed**: Optimizations needed for real-time applications

## Next Steps

After mastering this chapter, you can explore:

- **Advanced Transformer Variants**: Longformer, BigBird, Performer
- **Efficient Attention**: Linear attention, sparse attention
- **Multimodal Transformers**: Vision transformers, audio transformers
- **Large Language Models**: GPT-3, PaLM, LLaMA
- **Model Serving**: Production deployment of transformer models
- **Research Frontiers**: Latest developments in transformer architecture

## Resources

- **Original Transformer Paper**: "Attention Is All You Need"
- **BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers"
- **Hugging Face Transformers**: Popular transformer library
- **TensorFlow/PyTorch**: Deep learning frameworks with transformer support
- **Papers With Code**: Latest transformer research papers

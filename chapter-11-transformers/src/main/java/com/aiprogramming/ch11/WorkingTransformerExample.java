package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.List;
import java.util.ArrayList;

/**
 * Working transformer example demonstrating key concepts
 */
public class WorkingTransformerExample {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 11: Advanced NLP with Transformers (Working Example) ===\n");
        
        // Example 1: Basic Attention
        demonstrateBasicAttention();
        
        // Example 2: Positional Encoding
        demonstratePositionalEncoding();
        
        // Example 3: Tokenization
        demonstrateTokenization();
        
        // Example 4: Attention Masks
        demonstrateAttentionMasks();
        
        System.out.println("\n=== All working examples completed successfully! ===");
    }
    
    /**
     * Demonstrate basic attention mechanism
     */
    private static void demonstrateBasicAttention() {
        System.out.println("1. Basic Attention Mechanism");
        System.out.println("============================");
        
        // Create scaled dot-product attention
        Attention.ScaledDotProductAttention attention = new Attention.ScaledDotProductAttention(4);
        
        // Create simple 2x2 matrices
        RealMatrix queries = new Array2DRowRealMatrix(2, 2);
        RealMatrix keys = new Array2DRowRealMatrix(2, 2);
        RealMatrix values = new Array2DRowRealMatrix(2, 2);
        
        // Fill with simple values
        queries.setEntry(0, 0, 1.0); queries.setEntry(0, 1, 2.0);
        queries.setEntry(1, 0, 3.0); queries.setEntry(1, 1, 4.0);
        
        keys.setEntry(0, 0, 0.5); keys.setEntry(0, 1, 1.5);
        keys.setEntry(1, 0, 2.5); keys.setEntry(1, 1, 3.5);
        
        values.setEntry(0, 0, 0.1); values.setEntry(0, 1, 0.2);
        values.setEntry(1, 0, 0.3); values.setEntry(1, 1, 0.4);
        
        // Compute attention
        Attention.AttentionOutput output = attention.compute(queries, keys, values, null);
        
        System.out.println("Attention output shape: " + output.output.getRowDimension() + " x " + output.output.getColumnDimension());
        System.out.println("Attention weights shape: " + output.attentionWeights.getRowDimension() + " x " + output.attentionWeights.getColumnDimension());
        System.out.println("Sample attention output:");
        for (int i = 0; i < output.output.getRowDimension(); i++) {
            for (int j = 0; j < output.output.getColumnDimension(); j++) {
                System.out.printf("  [%d,%d]: %.4f\n", i, j, output.output.getEntry(i, j));
            }
        }
        System.out.println();
    }
    
    /**
     * Demonstrate positional encoding
     */
    private static void demonstratePositionalEncoding() {
        System.out.println("2. Positional Encoding");
        System.out.println("======================");
        
        // Create positional encoding
        int dModel = 4;
        int maxSeqLen = 6;
        TransformerLayer.PositionalEncoding posEncoding = new TransformerLayer.PositionalEncoding(dModel, maxSeqLen);
        
        // Get encoding for sequence length 3
        RealMatrix encoding = posEncoding.getEncoding(3);
        
        System.out.println("Positional encoding shape: " + encoding.getRowDimension() + " x " + encoding.getColumnDimension());
        System.out.println("Positional encoding values:");
        for (int i = 0; i < encoding.getRowDimension(); i++) {
            for (int j = 0; j < encoding.getColumnDimension(); j++) {
                System.out.printf("  Position %d, Dim %d: %.4f\n", i, j, encoding.getEntry(i, j));
            }
        }
        System.out.println();
    }
    
    /**
     * Demonstrate tokenization
     */
    private static void demonstrateTokenization() {
        System.out.println("3. Text Tokenization");
        System.out.println("====================");
        
        // Create tokenizer
        int vocabSize = 1000;
        int maxLength = 20;
        Tokenizer tokenizer = new Tokenizer(vocabSize, maxLength);
        
        // Sample text
        String text = "Hello world! This is a test.";
        
        System.out.println("Input text: " + text);
        
        // Tokenize
        List<String> tokens = tokenizer.tokenize(text);
        System.out.println("Tokenized: " + tokens);
        
        // Encode
        List<Integer> tokenIds = tokenizer.encode(tokens);
        System.out.println("Token IDs: " + tokenIds);
        
        // Decode
        List<String> decodedTokens = tokenizer.decode(tokenIds);
        System.out.println("Decoded: " + decodedTokens);
        
        // BERT encoding
        List<Integer> bertIds = tokenizer.encodeBERT(text);
        System.out.println("BERT IDs (first 10): " + bertIds.subList(0, Math.min(10, bertIds.size())));
        System.out.println();
    }
    
    /**
     * Demonstrate attention masks
     */
    private static void demonstrateAttentionMasks() {
        System.out.println("4. Attention Masks");
        System.out.println("==================");
        
        // Create causal mask
        int seqLen = 4;
        RealMatrix causalMask = Attention.MultiHeadAttention.createCausalMask(seqLen);
        
        System.out.println("Causal mask (" + seqLen + "x" + seqLen + "):");
        for (int i = 0; i < causalMask.getRowDimension(); i++) {
            for (int j = 0; j < causalMask.getColumnDimension(); j++) {
                double value = causalMask.getEntry(i, j);
                String display = (value == Double.NEGATIVE_INFINITY) ? "-inf" : String.format("%.1f", value);
                System.out.printf("  %s", display);
            }
            System.out.println();
        }
        System.out.println();
        
        // Create padding mask
        boolean[] isPadding = {false, false, true, true};
        RealMatrix paddingMask = Attention.MultiHeadAttention.createPaddingMask(isPadding);
        
        System.out.println("Padding mask:");
        for (int i = 0; i < paddingMask.getRowDimension(); i++) {
            for (int j = 0; j < paddingMask.getColumnDimension(); j++) {
                double value = paddingMask.getEntry(i, j);
                String display = (value == Double.NEGATIVE_INFINITY) ? "-inf" : String.format("%.1f", value);
                System.out.printf("  %s", display);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    /**
     * Demonstrate BERT concepts
     */
    public static void demonstrateBERTConcepts() {
        System.out.println("5. BERT Concepts");
        System.out.println("================");
        
        // Create tokenizer
        Tokenizer tokenizer = new Tokenizer(1000, 32);
        
        // Sample sentence pair
        String sentenceA = "The cat sat on the mat.";
        String sentenceB = "It was a sunny day.";
        
        System.out.println("Sentence A: " + sentenceA);
        System.out.println("Sentence B: " + sentenceB);
        
        // Encode sentence pair
        List<Integer> inputIds = tokenizer.encodeSentencePair(sentenceA, sentenceB);
        List<Integer> segmentIds = tokenizer.createSegmentIds(sentenceA, sentenceB);
        List<Integer> attentionMask = tokenizer.createAttentionMask(inputIds);
        
        System.out.println("Input IDs (first 15): " + inputIds.subList(0, Math.min(15, inputIds.size())));
        System.out.println("Segment IDs (first 15): " + segmentIds.subList(0, Math.min(15, segmentIds.size())));
        System.out.println("Attention mask (first 15): " + attentionMask.subList(0, Math.min(15, attentionMask.size())));
        
        // Show special tokens
        System.out.println("\nSpecial tokens:");
        System.out.println("  CLS token ID: " + tokenizer.getVocab().get(Tokenizer.CLS_TOKEN));
        System.out.println("  SEP token ID: " + tokenizer.getVocab().get(Tokenizer.SEP_TOKEN));
        System.out.println("  PAD token ID: " + tokenizer.getVocab().get(Tokenizer.PAD_TOKEN));
        System.out.println("  MASK token ID: " + tokenizer.getVocab().get(Tokenizer.MASK_TOKEN));
        System.out.println();
    }
}

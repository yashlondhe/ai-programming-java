package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.List;
import java.util.ArrayList;

/**
 * Simplified transformer example demonstrating key concepts
 */
public class SimpleTransformerExample {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 11: Advanced NLP with Transformers (Simplified) ===\n");
        
        // Example 1: Attention Mechanism
        demonstrateAttention();
        
        // Example 2: Multi-Head Attention
        demonstrateMultiHeadAttention();
        
        // Example 3: Positional Encoding
        demonstratePositionalEncoding();
        
        // Example 4: BERT Tokenization
        demonstrateBERTTokenization();
        
        System.out.println("\n=== All simplified examples completed successfully! ===");
    }
    
    /**
     * Demonstrate basic attention mechanism
     */
    private static void demonstrateAttention() {
        System.out.println("1. Attention Mechanism Example");
        System.out.println("==============================");
        
        // Create scaled dot-product attention
        Attention.ScaledDotProductAttention attention = new Attention.ScaledDotProductAttention(8);
        
        // Create sample input matrices
        RealMatrix queries = new Array2DRowRealMatrix(2, 8);
        RealMatrix keys = new Array2DRowRealMatrix(2, 8);
        RealMatrix values = new Array2DRowRealMatrix(2, 8);
        
        // Fill with sample data
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                queries.setEntry(i, j, Math.sin(i + j * 0.5));
                keys.setEntry(i, j, Math.cos(i + j * 0.3));
                values.setEntry(i, j, (i + j) * 0.1);
            }
        }
        
        // Compute attention
        Attention.AttentionOutput output = attention.compute(queries, keys, values, null);
        
        System.out.println("Attention output shape: " + output.output.getRowDimension() + " x " + output.output.getColumnDimension());
        System.out.println("Attention weights shape: " + output.attentionWeights.getRowDimension() + " x " + output.attentionWeights.getColumnDimension());
        System.out.println("Sample attention output values:");
        for (int i = 0; i < Math.min(3, output.output.getColumnDimension()); i++) {
            System.out.printf("  Position %d: %.4f\n", i, output.output.getEntry(0, i));
        }
        System.out.println();
    }
    
    /**
     * Demonstrate multi-head attention
     */
    private static void demonstrateMultiHeadAttention() {
        System.out.println("2. Multi-Head Attention Example");
        System.out.println("===============================");
        
        // Create multi-head attention
        int numHeads = 4;
        int dModel = 16;
        Attention.MultiHeadAttention attention = new Attention.MultiHeadAttention(numHeads, dModel);
        
        // Create sample input
        RealMatrix queries = new Array2DRowRealMatrix(1, dModel);
        RealMatrix keys = new Array2DRowRealMatrix(1, dModel);
        RealMatrix values = new Array2DRowRealMatrix(1, dModel);
        
        // Fill with sample data
        for (int i = 0; i < dModel; i++) {
            queries.setEntry(0, i, i * 0.5);
            keys.setEntry(0, i, i * 0.3);
            values.setEntry(0, i, i * 0.1);
        }
        
        // Compute multi-head attention
        RealMatrix output = attention.forward(queries, keys, values, null);
        
        System.out.println("Multi-head attention output shape: " + output.getRowDimension() + " x " + output.getColumnDimension());
        System.out.println("Sample output values:");
        for (int i = 0; i < Math.min(4, output.getColumnDimension()); i++) {
            System.out.printf("  Position %d: %.4f\n", i, output.getEntry(0, i));
        }
        System.out.println();
    }
    
    /**
     * Demonstrate positional encoding
     */
    private static void demonstratePositionalEncoding() {
        System.out.println("3. Positional Encoding Example");
        System.out.println("==============================");
        
        // Create positional encoding
        int dModel = 8;
        int maxSeqLen = 10;
        TransformerLayer.PositionalEncoding posEncoding = new TransformerLayer.PositionalEncoding(dModel, maxSeqLen);
        
        // Get encoding for sequence length 5
        RealMatrix encoding = posEncoding.getEncoding(5);
        
        System.out.println("Positional encoding shape: " + encoding.getRowDimension() + " x " + encoding.getColumnDimension());
        System.out.println("Sample positional encoding values:");
        for (int i = 0; i < Math.min(3, encoding.getRowDimension()); i++) {
            for (int j = 0; j < Math.min(4, encoding.getColumnDimension()); j++) {
                System.out.printf("  Position %d, Dim %d: %.4f\n", i, j, encoding.getEntry(i, j));
            }
        }
        System.out.println();
    }
    
    /**
     * Demonstrate BERT tokenization
     */
    private static void demonstrateBERTTokenization() {
        System.out.println("4. BERT Tokenization Example");
        System.out.println("============================");
        
        // Create tokenizer
        int vocabSize = 1000;
        int maxLength = 32;
        Tokenizer tokenizer = new Tokenizer(vocabSize, maxLength);
        
        // Sample text
        String sentenceA = "The cat sat on the mat.";
        String sentenceB = "It was a sunny day.";
        
        System.out.println("Input sentences:");
        System.out.println("  A: " + sentenceA);
        System.out.println("  B: " + sentenceB);
        
        // Tokenize
        List<String> tokensA = tokenizer.tokenize(sentenceA);
        List<String> tokensB = tokenizer.tokenize(sentenceB);
        
        System.out.println("Tokenized sentence A: " + tokensA);
        System.out.println("Tokenized sentence B: " + tokensB);
        
        // Encode for BERT
        List<Integer> inputIds = tokenizer.encodeSentencePair(sentenceA, sentenceB);
        List<Integer> segmentIds = tokenizer.createSegmentIds(sentenceA, sentenceB);
        List<Integer> attentionMask = tokenizer.createAttentionMask(inputIds);
        
        System.out.println("Input IDs (first 10): " + inputIds.subList(0, Math.min(10, inputIds.size())));
        System.out.println("Segment IDs (first 10): " + segmentIds.subList(0, Math.min(10, segmentIds.size())));
        System.out.println("Attention mask (first 10): " + attentionMask.subList(0, Math.min(10, attentionMask.size())));
        System.out.println();
    }
    
    /**
     * Demonstrate attention mask creation
     */
    public static void demonstrateAttentionMasks() {
        System.out.println("5. Attention Masks Example");
        System.out.println("==========================");
        
        // Create causal mask
        int seqLen = 6;
        RealMatrix causalMask = Attention.MultiHeadAttention.createCausalMask(seqLen);
        
        System.out.println("Causal mask shape: " + causalMask.getRowDimension() + " x " + causalMask.getColumnDimension());
        System.out.println("Causal mask (showing -inf as -999):");
        for (int i = 0; i < Math.min(4, causalMask.getRowDimension()); i++) {
            for (int j = 0; j < Math.min(4, causalMask.getColumnDimension()); j++) {
                double value = causalMask.getEntry(i, j);
                String display = (value == Double.NEGATIVE_INFINITY) ? "-999" : String.format("%.1f", value);
                System.out.printf("  %s", display);
            }
            System.out.println();
        }
        System.out.println();
        
        // Create padding mask
        boolean[] isPadding = {false, false, false, true, true, true};
        RealMatrix paddingMask = Attention.MultiHeadAttention.createPaddingMask(isPadding);
        
        System.out.println("Padding mask shape: " + paddingMask.getRowDimension() + " x " + paddingMask.getColumnDimension());
        System.out.println("Padding mask (showing -inf as -999):");
        for (int i = 0; i < Math.min(4, paddingMask.getRowDimension()); i++) {
            for (int j = 0; j < Math.min(4, paddingMask.getColumnDimension()); j++) {
                double value = paddingMask.getEntry(i, j);
                String display = (value == Double.NEGATIVE_INFINITY) ? "-999" : String.format("%.1f", value);
                System.out.printf("  %s", display);
            }
            System.out.println();
        }
        System.out.println();
    }
}

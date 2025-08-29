package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.List;
import java.util.ArrayList;

/**
 * Main example demonstrating transformer and BERT usage
 */
public class TransformerExample {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 11: Advanced NLP with Transformers ===\n");
        
        // Example 1: Basic Transformer
        demonstrateBasicTransformer();
        
        // Example 2: BERT Model
        demonstrateBERT();
        
        // Example 3: Text Generation
        demonstrateTextGeneration();
        
        // Example 4: Question Answering
        demonstrateQuestionAnswering();
        
        System.out.println("\n=== All examples completed successfully! ===");
    }
    
    /**
     * Demonstrate basic transformer functionality
     */
    private static void demonstrateBasicTransformer() {
        System.out.println("1. Basic Transformer Example");
        System.out.println("============================");
        
        // Create a small transformer
        int dModel = 32;
        int numLayers = 2;
        int numHeads = 4;
        int dff = 64;
        int vocabSize = 1000;
        int maxSeqLen = 20;
        
        Transformer transformer = new Transformer(dModel, numLayers, numHeads, dff, vocabSize, maxSeqLen);
        
        // Create sample input
        RealMatrix encoderInput = new Array2DRowRealMatrix(1, 5);
        RealMatrix decoderInput = new Array2DRowRealMatrix(1, 4);
        
        // Fill with random token IDs
        for (int i = 0; i < 5; i++) {
            encoderInput.setEntry(0, i, (int)(Math.random() * vocabSize));
        }
        for (int i = 0; i < 4; i++) {
            decoderInput.setEntry(0, i, (int)(Math.random() * vocabSize));
        }
        
        // Create masks
        RealMatrix encoderMask = new Array2DRowRealMatrix(5, 5);
        RealMatrix decoderMask = Attention.MultiHeadAttention.createCausalMask(4);
        
        // Forward pass
        RealMatrix output = transformer.forward(encoderInput, decoderInput, encoderMask, decoderMask);
        
        System.out.println("Transformer output shape: " + output.getRowDimension() + " x " + output.getColumnDimension());
        System.out.println("Sample output values:");
        for (int i = 0; i < Math.min(5, output.getColumnDimension()); i++) {
            System.out.printf("  Position %d: %.4f\n", i, output.getEntry(0, i));
        }
        System.out.println();
    }
    
    /**
     * Demonstrate BERT model functionality
     */
    private static void demonstrateBERT() {
        System.out.println("2. BERT Model Example");
        System.out.println("=====================");
        
        // Create BERT model
        int dModel = 128;
        int numLayers = 2;
        int numHeads = 4;
        int dff = 256;
        int vocabSize = 1000;
        int maxSeqLen = 64;
        
        BERT bert = new BERT(dModel, numLayers, numHeads, dff, vocabSize, maxSeqLen);
        
        // Create tokenizer
        Tokenizer tokenizer = new Tokenizer(vocabSize, maxSeqLen);
        
        // Sample text
        String sentenceA = "The cat sat on the mat.";
        String sentenceB = "It was a sunny day.";
        
        // Encode sentences
        List<Integer> inputIds = tokenizer.encodeSentencePair(sentenceA, sentenceB);
        List<Integer> segmentIds = tokenizer.createSegmentIds(sentenceA, sentenceB);
        List<Integer> attentionMask = tokenizer.createAttentionMask(inputIds);
        
        // Convert to matrices
        RealMatrix inputMatrix = new Array2DRowRealMatrix(1, inputIds.size());
        RealMatrix segmentMatrix = new Array2DRowRealMatrix(1, segmentIds.size());
        RealMatrix maskMatrix = new Array2DRowRealMatrix(attentionMask.size(), attentionMask.size());
        
        for (int i = 0; i < inputIds.size(); i++) {
            inputMatrix.setEntry(0, i, inputIds.get(i));
            segmentMatrix.setEntry(0, i, segmentIds.get(i));
        }
        
        for (int i = 0; i < attentionMask.size(); i++) {
            for (int j = 0; j < attentionMask.size(); j++) {
                if (attentionMask.get(i) == 0 || attentionMask.get(j) == 0) {
                    maskMatrix.setEntry(i, j, Double.NEGATIVE_INFINITY);
                }
            }
        }
        
        // Forward pass
        BERT.BERTOutput output = bert.forward(inputMatrix, segmentMatrix, maskMatrix);
        
        System.out.println("Input sentences:");
        System.out.println("  A: " + sentenceA);
        System.out.println("  B: " + sentenceB);
        System.out.println("BERT hidden states shape: " + output.hiddenStates.getRowDimension() + 
                          " x " + output.hiddenStates.getColumnDimension());
        System.out.println("MLM logits shape: " + output.mlmLogits.getRowDimension() + 
                          " x " + output.mlmLogits.getColumnDimension());
        System.out.println("NSP logits shape: " + output.nspLogits.getRowDimension() + 
                          " x " + output.nspLogits.getColumnDimension());
        System.out.println();
    }
    
    /**
     * Demonstrate text generation with transformer
     */
    private static void demonstrateTextGeneration() {
        System.out.println("3. Text Generation Example");
        System.out.println("==========================");
        
        // Create a small transformer for generation
        int dModel = 64;
        int numLayers = 2;
        int numHeads = 4;
        int dff = 128;
        int vocabSize = 1000;
        int maxSeqLen = 50;
        
        Transformer transformer = new Transformer(dModel, numLayers, numHeads, dff, vocabSize, maxSeqLen);
        
        // Create encoder input (context)
        RealMatrix encoderInput = new Array2DRowRealMatrix(1, 5);
        for (int i = 0; i < 5; i++) {
            encoderInput.setEntry(0, i, 100 + i); // Some context tokens
        }
        
        // Create encoder mask
        RealMatrix encoderMask = new Array2DRowRealMatrix(5, 5);
        
        // Generate text
        int startToken = 101; // Start token
        int endToken = 102;   // End token
        int maxLength = 10;
        
        List<Integer> generated = transformer.generate(encoderInput, encoderMask, maxLength, startToken, endToken);
        
        System.out.println("Generated token sequence:");
        System.out.print("  ");
        for (int token : generated) {
            System.out.print(token + " ");
        }
        System.out.println();
        System.out.println("Generation completed with " + generated.size() + " tokens.");
        System.out.println();
    }
    
    /**
     * Demonstrate question answering with BERT
     */
    private static void demonstrateQuestionAnswering() {
        System.out.println("4. Question Answering Example");
        System.out.println("=============================");
        
        // Create BERT model
        int dModel = 128;
        int numLayers = 2;
        int numHeads = 4;
        int dff = 256;
        int vocabSize = 1000;
        int maxSeqLen = 64;
        
        BERT bert = new BERT(dModel, numLayers, numHeads, dff, vocabSize, maxSeqLen);
        
        // Create tokenizer
        Tokenizer tokenizer = new Tokenizer(vocabSize, maxSeqLen);
        
        // Question answering example
        String question = "What is the capital of France?";
        String context = "Paris is the capital of France. It is a beautiful city.";
        
        // Encode question and context
        List<Integer> inputIds = tokenizer.encodeSentencePair(question, context);
        List<Integer> segmentIds = tokenizer.createSegmentIds(question, context);
        List<Integer> attentionMask = tokenizer.createAttentionMask(inputIds);
        
        // Convert to matrices
        RealMatrix inputMatrix = new Array2DRowRealMatrix(1, inputIds.size());
        RealMatrix segmentMatrix = new Array2DRowRealMatrix(1, segmentIds.size());
        RealMatrix maskMatrix = new Array2DRowRealMatrix(attentionMask.size(), attentionMask.size());
        
        for (int i = 0; i < inputIds.size(); i++) {
            inputMatrix.setEntry(0, i, inputIds.get(i));
            segmentMatrix.setEntry(0, i, segmentIds.get(i));
        }
        
        for (int i = 0; i < attentionMask.size(); i++) {
            for (int j = 0; j < attentionMask.size(); j++) {
                if (attentionMask.get(i) == 0 || attentionMask.get(j) == 0) {
                    maskMatrix.setEntry(i, j, Double.NEGATIVE_INFINITY);
                }
            }
        }
        
        // Get BERT representations
        BERT.BERTOutput output = bert.forward(inputMatrix, segmentMatrix, maskMatrix);
        
        System.out.println("Question: " + question);
        System.out.println("Context: " + context);
        System.out.println("BERT hidden states shape: " + output.hiddenStates.getRowDimension() + 
                          " x " + output.hiddenStates.getColumnDimension());
        
        // Simple answer extraction (in practice, you'd use a more sophisticated approach)
        System.out.println("Answer extraction: The model can now use the hidden states to");
        System.out.println("identify the answer span in the context.");
        System.out.println();
    }
    
    /**
     * Demonstrate attention visualization
     */
    public static void demonstrateAttention() {
        System.out.println("5. Attention Visualization");
        System.out.println("=========================");
        
        // Create multi-head attention
        int numHeads = 4;
        int dModel = 64;
        Attention.MultiHeadAttention attention = new Attention.MultiHeadAttention(numHeads, dModel);
        
        // Create sample input
        RealMatrix queries = new Array2DRowRealMatrix(1, 8);
        RealMatrix keys = new Array2DRowRealMatrix(1, 8);
        RealMatrix values = new Array2DRowRealMatrix(1, 8);
        
        // Fill with some values
        for (int i = 0; i < 8; i++) {
            queries.setEntry(0, i, Math.sin(i * 0.5));
            keys.setEntry(0, i, Math.cos(i * 0.3));
            values.setEntry(0, i, i * 0.1);
        }
        
        // Compute attention
        RealMatrix output = attention.forward(queries, keys, values, null);
        
        System.out.println("Multi-head attention output shape: " + output.getRowDimension() + 
                          " x " + output.getColumnDimension());
        System.out.println("Sample attention output values:");
        for (int i = 0; i < Math.min(5, output.getColumnDimension()); i++) {
            System.out.printf("  Position %d: %.4f\n", i, output.getEntry(0, i));
        }
        System.out.println();
    }
}

package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.List;
import java.util.ArrayList;

/**
 * Complete Transformer model implementation
 */
public class Transformer {
    private final int dModel;
    private final int numLayers;
    private final int numHeads;
    private final int dff;
    private final int vocabSize;
    private final int maxSeqLen;
    
    // Embedding layers
    private RealMatrix tokenEmbeddings;
    private TransformerLayer.PositionalEncoding positionalEncoding;
    
    // Encoder and Decoder layers
    private List<TransformerLayer.EncoderLayer> encoderLayers;
    private List<TransformerLayer.DecoderLayer> decoderLayers;
    
    // Output projection
    private RealMatrix outputProjection;
    
    public Transformer(int dModel, int numLayers, int numHeads, int dff, 
                      int vocabSize, int maxSeqLen) {
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dff = dff;
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        
        initializeModel();
    }
    
    private void initializeModel() {
        // Initialize token embeddings
        tokenEmbeddings = new Array2DRowRealMatrix(vocabSize, dModel);
        initializeEmbeddings();
        
        // Initialize positional encoding
        positionalEncoding = new TransformerLayer.PositionalEncoding(dModel, maxSeqLen);
        
        // Initialize encoder layers
        encoderLayers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            encoderLayers.add(new TransformerLayer.EncoderLayer(dModel, numHeads, dff));
        }
        
        // Initialize decoder layers
        decoderLayers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            decoderLayers.add(new TransformerLayer.DecoderLayer(dModel, numHeads, dff));
        }
        
        // Initialize output projection
        outputProjection = new Array2DRowRealMatrix(dModel, vocabSize);
        initializeOutputProjection();
    }
    
    private void initializeEmbeddings() {
        // Simple random initialization for embeddings
        java.util.Random random = new java.util.Random(42);
        double scale = Math.sqrt(2.0 / dModel);
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < dModel; j++) {
                tokenEmbeddings.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
    }
    
    private void initializeOutputProjection() {
        java.util.Random random = new java.util.Random(42);
        double scale = Math.sqrt(2.0 / dModel);
        
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSize; j++) {
                outputProjection.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
    }
    
    /**
     * Forward pass through the complete transformer
     */
    public RealMatrix forward(RealMatrix encoderInput, RealMatrix decoderInput, 
                            RealMatrix encoderMask, RealMatrix decoderMask) {
        // Encoder forward pass
        RealMatrix encoderOutput = encode(encoderInput, encoderMask);
        
        // Decoder forward pass
        RealMatrix decoderOutput = decode(decoderInput, encoderOutput, decoderMask, encoderMask);
        
        // Output projection
        RealMatrix logits = decoderOutput.multiply(outputProjection);
        
        return logits;
    }
    
    /**
     * Encoder forward pass
     */
    private RealMatrix encode(RealMatrix input, RealMatrix mask) {
        // Token embedding + positional encoding
        RealMatrix embedded = embedTokens(input);
        RealMatrix posEncoded = addPositionalEncoding(embedded);
        
        // Pass through encoder layers
        RealMatrix output = posEncoded;
        for (TransformerLayer.EncoderLayer layer : encoderLayers) {
            output = layer.forward(output, mask);
        }
        
        return output;
    }
    
    /**
     * Decoder forward pass
     */
    private RealMatrix decode(RealMatrix input, RealMatrix encoderOutput, 
                            RealMatrix lookAheadMask, RealMatrix paddingMask) {
        // Token embedding + positional encoding
        RealMatrix embedded = embedTokens(input);
        RealMatrix posEncoded = addPositionalEncoding(embedded);
        
        // Pass through decoder layers
        RealMatrix output = posEncoded;
        for (TransformerLayer.DecoderLayer layer : decoderLayers) {
            output = layer.forward(output, encoderOutput, lookAheadMask, paddingMask);
        }
        
        return output;
    }
    
    /**
     * Convert token indices to embeddings
     */
    private RealMatrix embedTokens(RealMatrix tokenIndices) {
        int batchSize = tokenIndices.getRowDimension();
        int seqLen = tokenIndices.getColumnDimension();
        RealMatrix embeddings = new Array2DRowRealMatrix(batchSize, seqLen);
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                int tokenId = (int) tokenIndices.getEntry(i, j);
                if (tokenId >= 0 && tokenId < vocabSize) {
                    // For simplicity, we'll use the token ID as a simple embedding
                    // In a real implementation, you'd use the full embedding matrix
                    embeddings.setEntry(i, j, tokenId);
                }
            }
        }
        
        return embeddings;
    }
    
    /**
     * Add positional encoding to embeddings
     */
    private RealMatrix addPositionalEncoding(RealMatrix embeddings) {
        int seqLen = embeddings.getColumnDimension();
        RealMatrix posEncoding = positionalEncoding.getEncoding(seqLen);
        
        // For simplicity, we'll just return the embeddings with positional encoding added
        // In a real implementation, you'd properly add the positional encoding
        RealMatrix result = new Array2DRowRealMatrix(embeddings.getRowDimension(), seqLen);
        for (int i = 0; i < embeddings.getRowDimension(); i++) {
            for (int j = 0; j < seqLen; j++) {
                result.setEntry(i, j, embeddings.getEntry(i, j) + posEncoding.getEntry(j, 0));
            }
        }
        return result;
    }
    
    /**
     * Generate text using the transformer (autoregressive generation)
     */
    public List<Integer> generate(RealMatrix encoderInput, RealMatrix encoderMask, 
                                int maxLength, int startToken, int endToken) {
        List<Integer> generated = new ArrayList<>();
        generated.add(startToken);
        
        RealMatrix encoderOutput = encode(encoderInput, encoderMask);
        
        for (int i = 0; i < maxLength; i++) {
            // Create decoder input from generated tokens
            RealMatrix decoderInput = new Array2DRowRealMatrix(1, generated.size());
            for (int j = 0; j < generated.size(); j++) {
                decoderInput.setEntry(0, j, generated.get(j));
            }
            
            // Create causal mask for decoder
            RealMatrix lookAheadMask = Attention.MultiHeadAttention.createCausalMask(generated.size());
            
            // Forward pass
            RealMatrix logits = forward(encoderInput, decoderInput, encoderMask, lookAheadMask);
            
            // Get next token (greedy decoding)
            int nextToken = getNextToken(logits);
            generated.add(nextToken);
            
            // Stop if end token is generated
            if (nextToken == endToken) {
                break;
            }
        }
        
        return generated;
    }
    
    /**
     * Get next token using greedy decoding
     */
    private int getNextToken(RealMatrix logits) {
        // Get the last position's logits
        int lastPos = logits.getColumnDimension() - 1;
        double maxLogit = Double.NEGATIVE_INFINITY;
        int bestToken = 0;
        
        for (int i = 0; i < vocabSize; i++) {
            double logit = logits.getEntry(0, i);
            if (logit > maxLogit) {
                maxLogit = logit;
                bestToken = i;
            }
        }
        
        return bestToken;
    }
    
    /**
     * Get model parameters for training
     */
    public List<RealMatrix> getParameters() {
        List<RealMatrix> params = new ArrayList<>();
        params.add(tokenEmbeddings);
        params.add(outputProjection);
        // Note: In a full implementation, you would also add parameters from encoder/decoder layers
        return params;
    }
    
    /**
     * Set model parameters (for loading pre-trained models)
     */
    public void setParameters(List<RealMatrix> params) {
        if (params.size() >= 2) {
            this.tokenEmbeddings = params.get(0);
            this.outputProjection = params.get(1);
        }
    }
}

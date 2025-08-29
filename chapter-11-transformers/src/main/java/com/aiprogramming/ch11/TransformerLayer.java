package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.Random;

/**
 * Transformer layer implementations (Encoder and Decoder)
 */
public class TransformerLayer {
    
    /**
     * Feed-Forward Network used in transformer layers
     */
    public static class FeedForwardNetwork {
        private final int dModel;
        private final int dff;
        private RealMatrix W1; // First linear layer weights
        private RealMatrix b1; // First linear layer bias
        private RealMatrix W2; // Second linear layer weights
        private RealMatrix b2; // Second linear layer bias
        
        public FeedForwardNetwork(int dModel, int dff) {
            this.dModel = dModel;
            this.dff = dff;
            initializeWeights();
        }
        
        private void initializeWeights() {
            Random random = new Random(42);
            double scale1 = Math.sqrt(2.0 / dModel);
            double scale2 = Math.sqrt(2.0 / dff);
            
            W1 = new Array2DRowRealMatrix(dff, dModel);
            b1 = new Array2DRowRealMatrix(dff, 1);
            W2 = new Array2DRowRealMatrix(dModel, dff);
            b2 = new Array2DRowRealMatrix(dModel, 1);
            
            // Initialize weights
            for (int i = 0; i < dff; i++) {
                for (int j = 0; j < dModel; j++) {
                    W1.setEntry(i, j, random.nextGaussian() * scale1);
                }
                b1.setEntry(i, 0, 0.0);
            }
            
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dff; j++) {
                    W2.setEntry(i, j, random.nextGaussian() * scale2);
                }
                b2.setEntry(i, 0, 0.0);
            }
        }
        
        /**
         * Forward pass through the feed-forward network
         */
        public RealMatrix forward(RealMatrix input) {
            // First linear transformation + ReLU
            RealMatrix hidden = W1.multiply(input.transpose()).add(b1);
            hidden = relu(hidden);
            
            // Second linear transformation
            RealMatrix output = W2.multiply(hidden).add(b2);
            
            return output.transpose();
        }
        
        /**
         * ReLU activation function
         */
        private RealMatrix relu(RealMatrix matrix) {
            RealMatrix result = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
            for (int i = 0; i < matrix.getRowDimension(); i++) {
                for (int j = 0; j < matrix.getColumnDimension(); j++) {
                    result.setEntry(i, j, Math.max(0, matrix.getEntry(i, j)));
                }
            }
            return result;
        }
    }
    
    /**
     * Layer Normalization implementation
     */
    public static class LayerNormalization {
        private final int dModel;
        private RealMatrix gamma; // Scale parameter
        private RealMatrix beta;  // Shift parameter
        
        public LayerNormalization(int dModel) {
            this.dModel = dModel;
            initializeParameters();
        }
        
        private void initializeParameters() {
            gamma = new Array2DRowRealMatrix(dModel, 1);
            beta = new Array2DRowRealMatrix(dModel, 1);
            
            // Initialize gamma to 1 and beta to 0
            for (int i = 0; i < dModel; i++) {
                gamma.setEntry(i, 0, 1.0);
                beta.setEntry(i, 0, 0.0);
            }
        }
        
        /**
         * Apply layer normalization
         */
        public RealMatrix normalize(RealMatrix input) {
            int batchSize = input.getRowDimension();
            int seqLen = input.getColumnDimension();
            RealMatrix output = new Array2DRowRealMatrix(batchSize, seqLen);
            
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < seqLen; j++) {
                    // Compute mean and variance for this position
                    double mean = 0.0;
                    double variance = 0.0;
                    
                    for (int k = 0; k < dModel; k++) {
                        mean += input.getEntry(i, k);
                    }
                    mean /= dModel;
                    
                    for (int k = 0; k < dModel; k++) {
                        double diff = input.getEntry(i, k) - mean;
                        variance += diff * diff;
                    }
                    variance /= dModel;
                    
                    // Normalize and apply scale/shift
                    double normalized = (input.getEntry(i, j) - mean) / Math.sqrt(variance + 1e-6);
                    output.setEntry(i, j, gamma.getEntry(j, 0) * normalized + beta.getEntry(j, 0));
                }
            }
            
            return output;
        }
    }
    
    /**
     * Transformer Encoder Layer
     */
    public static class EncoderLayer {
        private final Attention.MultiHeadAttention selfAttention;
        private final FeedForwardNetwork feedForward;
        private final LayerNormalization layerNorm1;
        private final LayerNormalization layerNorm2;
        private final int dModel;
        
        public EncoderLayer(int dModel, int numHeads, int dff) {
            this.dModel = dModel;
            this.selfAttention = new Attention.MultiHeadAttention(numHeads, dModel);
            this.feedForward = new FeedForwardNetwork(dModel, dff);
            this.layerNorm1 = new LayerNormalization(dModel);
            this.layerNorm2 = new LayerNormalization(dModel);
        }
        
        /**
         * Forward pass through encoder layer
         */
        public RealMatrix forward(RealMatrix input, RealMatrix mask) {
            // Self-attention with residual connection and layer norm
            RealMatrix attentionOutput = selfAttention.forward(input, input, input, mask);
            RealMatrix residual1 = input.add(attentionOutput);
            RealMatrix normalized1 = layerNorm1.normalize(residual1);
            
            // Feed-forward with residual connection and layer norm
            RealMatrix ffOutput = feedForward.forward(normalized1);
            RealMatrix residual2 = normalized1.add(ffOutput);
            RealMatrix output = layerNorm2.normalize(residual2);
            
            return output;
        }
    }
    
    /**
     * Transformer Decoder Layer
     */
    public static class DecoderLayer {
        private final Attention.MultiHeadAttention selfAttention;
        private final Attention.MultiHeadAttention crossAttention;
        private final FeedForwardNetwork feedForward;
        private final LayerNormalization layerNorm1;
        private final LayerNormalization layerNorm2;
        private final LayerNormalization layerNorm3;
        private final int dModel;
        
        public DecoderLayer(int dModel, int numHeads, int dff) {
            this.dModel = dModel;
            this.selfAttention = new Attention.MultiHeadAttention(numHeads, dModel);
            this.crossAttention = new Attention.MultiHeadAttention(numHeads, dModel);
            this.feedForward = new FeedForwardNetwork(dModel, dff);
            this.layerNorm1 = new LayerNormalization(dModel);
            this.layerNorm2 = new LayerNormalization(dModel);
            this.layerNorm3 = new LayerNormalization(dModel);
        }
        
        /**
         * Forward pass through decoder layer
         */
        public RealMatrix forward(RealMatrix input, RealMatrix encoderOutput, 
                                RealMatrix lookAheadMask, RealMatrix paddingMask) {
            // Self-attention with causal mask
            RealMatrix selfAttentionOutput = selfAttention.forward(input, input, input, lookAheadMask);
            RealMatrix residual1 = input.add(selfAttentionOutput);
            RealMatrix normalized1 = layerNorm1.normalize(residual1);
            
            // Cross-attention with encoder output
            RealMatrix crossAttentionOutput = crossAttention.forward(
                normalized1, encoderOutput, encoderOutput, paddingMask);
            RealMatrix residual2 = normalized1.add(crossAttentionOutput);
            RealMatrix normalized2 = layerNorm2.normalize(residual2);
            
            // Feed-forward
            RealMatrix ffOutput = feedForward.forward(normalized2);
            RealMatrix residual3 = normalized2.add(ffOutput);
            RealMatrix output = layerNorm3.normalize(residual3);
            
            return output;
        }
    }
    
    /**
     * Positional Encoding for transformer
     */
    public static class PositionalEncoding {
        private final int dModel;
        private final int maxSeqLen;
        private RealMatrix encoding;
        
        public PositionalEncoding(int dModel, int maxSeqLen) {
            this.dModel = dModel;
            this.maxSeqLen = maxSeqLen;
            this.encoding = new Array2DRowRealMatrix(maxSeqLen, dModel);
            generateEncoding();
        }
        
        private void generateEncoding() {
            for (int pos = 0; pos < maxSeqLen; pos++) {
                for (int i = 0; i < dModel; i++) {
                    if (i % 2 == 0) {
                        // Even indices: sin
                        encoding.setEntry(pos, i, Math.sin(pos / Math.pow(10000, i / (double) dModel)));
                    } else {
                        // Odd indices: cos
                        encoding.setEntry(pos, i, Math.cos(pos / Math.pow(10000, (i - 1) / (double) dModel)));
                    }
                }
            }
        }
        
        /**
         * Get positional encoding for a sequence
         */
        public RealMatrix getEncoding(int seqLen) {
            // Ensure sequence length doesn't exceed maxSeqLen
            int actualSeqLen = Math.min(seqLen, maxSeqLen);
            RealMatrix result = new Array2DRowRealMatrix(actualSeqLen, dModel);
            for (int i = 0; i < actualSeqLen; i++) {
                for (int j = 0; j < dModel; j++) {
                    result.setEntry(i, j, encoding.getEntry(i, j));
                }
            }
            return result;
        }
    }
}

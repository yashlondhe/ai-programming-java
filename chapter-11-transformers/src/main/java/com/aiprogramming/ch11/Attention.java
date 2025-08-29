package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import com.aiprogramming.utils.MatrixUtils;
import com.aiprogramming.utils.StatisticsUtils;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * Implementation of attention mechanisms for transformers
 */
public class Attention {
    
    /**
     * Scaled Dot-Product Attention implementation
     */
    public static class ScaledDotProductAttention {
        private final double scaleFactor;
        
        public ScaledDotProductAttention(int dk) {
            this.scaleFactor = Math.sqrt(dk);
        }
        
        /**
         * Compute scaled dot-product attention
         * @param queries Query matrix (batch_size, seq_len, dk)
         * @param keys Key matrix (batch_size, seq_len, dk)
         * @param values Value matrix (batch_size, seq_len, dv)
         * @param mask Optional attention mask
         * @return Attention output and attention weights
         */
        public AttentionOutput compute(RealMatrix queries, RealMatrix keys, RealMatrix values, RealMatrix mask) {
            // Compute attention scores: Q * K^T
            RealMatrix scores = queries.multiply(keys.transpose());
            
            // Scale the scores
            scores = scores.scalarMultiply(1.0 / scaleFactor);
            
            // Apply mask if provided
            if (mask != null) {
                scores = scores.add(mask);
            }
            
            // Apply softmax to get attention weights
            RealMatrix attentionWeights = softmax(scores);
            
            // Apply attention weights to values
            RealMatrix output = attentionWeights.multiply(values);
            
            return new AttentionOutput(output, attentionWeights);
        }
        
        /**
         * Apply softmax to attention scores
         */
        private RealMatrix softmax(RealMatrix scores) {
            int rows = scores.getRowDimension();
            int cols = scores.getColumnDimension();
            RealMatrix result = new Array2DRowRealMatrix(rows, cols);
            
            for (int i = 0; i < rows; i++) {
                double maxScore = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < cols; j++) {
                    maxScore = Math.max(maxScore, scores.getEntry(i, j));
                }
                
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    double expScore = Math.exp(scores.getEntry(i, j) - maxScore);
                    result.setEntry(i, j, expScore);
                    sum += expScore;
                }
                
                for (int j = 0; j < cols; j++) {
                    result.setEntry(i, j, result.getEntry(i, j) / sum);
                }
            }
            
            return result;
        }
    }
    
    /**
     * Multi-Head Attention implementation
     */
    public static class MultiHeadAttention {
        private final int numHeads;
        private final int dModel;
        private final int dk;
        private final int dv;
        private final ScaledDotProductAttention attention;
        
        // Learnable parameters
        private RealMatrix Wq; // Query weight matrix
        private RealMatrix Wk; // Key weight matrix
        private RealMatrix Wv; // Value weight matrix
        private RealMatrix Wo; // Output weight matrix
        
        public MultiHeadAttention(int numHeads, int dModel) {
            this.numHeads = numHeads;
            this.dModel = dModel;
            this.dk = dModel / numHeads;
            this.dv = dModel / numHeads;
            this.attention = new ScaledDotProductAttention(dk);
            
            initializeWeights();
        }
        
        private void initializeWeights() {
            Random random = new Random(42); // Fixed seed for reproducibility
            
            // Initialize weight matrices with Xavier/Glorot initialization
            double scale = Math.sqrt(2.0 / dModel);
            
            Wq = new Array2DRowRealMatrix(dModel, dModel);
            Wk = new Array2DRowRealMatrix(dModel, dModel);
            Wv = new Array2DRowRealMatrix(dModel, dModel);
            Wo = new Array2DRowRealMatrix(dModel, dModel);
            
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    Wq.setEntry(i, j, random.nextGaussian() * scale);
                    Wk.setEntry(i, j, random.nextGaussian() * scale);
                    Wv.setEntry(i, j, random.nextGaussian() * scale);
                    Wo.setEntry(i, j, random.nextGaussian() * scale);
                }
            }
        }
        
        /**
         * Forward pass of multi-head attention
         * @param queries Input queries
         * @param keys Input keys
         * @param values Input values
         * @param mask Optional attention mask
         * @return Attention output
         */
        public RealMatrix forward(RealMatrix queries, RealMatrix keys, RealMatrix values, RealMatrix mask) {
            int batchSize = queries.getRowDimension();
            int seqLen = queries.getColumnDimension();
            
            // Linear transformations
            RealMatrix Q = queries.multiply(Wq);
            RealMatrix K = keys.multiply(Wk);
            RealMatrix V = values.multiply(Wv);
            
            // Split into multiple heads
            List<RealMatrix> QHeads = splitHeads(Q, batchSize, seqLen);
            List<RealMatrix> KHeads = splitHeads(K, batchSize, seqLen);
            List<RealMatrix> VHeads = splitHeads(V, batchSize, seqLen);
            
            // Apply attention for each head
            List<RealMatrix> attentionOutputs = new ArrayList<>();
            for (int h = 0; h < numHeads; h++) {
                AttentionOutput output = attention.compute(QHeads.get(h), KHeads.get(h), VHeads.get(h), mask);
                attentionOutputs.add(output.output);
            }
            
            // Concatenate attention outputs
            RealMatrix concatenated = concatenateHeads(attentionOutputs, batchSize, seqLen);
            
            // Final linear transformation
            return concatenated.multiply(Wo);
        }
        
        /**
         * Split matrix into multiple attention heads
         */
        private List<RealMatrix> splitHeads(RealMatrix matrix, int batchSize, int seqLen) {
            List<RealMatrix> heads = new ArrayList<>();
            
            for (int h = 0; h < numHeads; h++) {
                RealMatrix head = new Array2DRowRealMatrix(batchSize, seqLen);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        head.setEntry(i, j, matrix.getEntry(i, j + h * dk));
                    }
                }
                heads.add(head);
            }
            
            return heads;
        }
        
        /**
         * Concatenate attention heads back into a single matrix
         */
        private RealMatrix concatenateHeads(List<RealMatrix> heads, int batchSize, int seqLen) {
            RealMatrix concatenated = new Array2DRowRealMatrix(batchSize, dModel);
            
            for (int h = 0; h < numHeads; h++) {
                RealMatrix head = heads.get(h);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        concatenated.setEntry(i, j + h * dk, head.getEntry(i, j));
                    }
                }
            }
            
            return concatenated;
        }
        
        /**
         * Create causal mask for decoder self-attention
         */
        public static RealMatrix createCausalMask(int seqLen) {
            RealMatrix mask = new Array2DRowRealMatrix(seqLen, seqLen);
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (j > i) {
                        mask.setEntry(i, j, Double.NEGATIVE_INFINITY);
                    }
                }
            }
            return mask;
        }
        
        /**
         * Create padding mask for variable length sequences
         */
        public static RealMatrix createPaddingMask(boolean[] isPadding) {
            int seqLen = isPadding.length;
            RealMatrix mask = new Array2DRowRealMatrix(seqLen, seqLen);
            
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (isPadding[i] || isPadding[j]) {
                        mask.setEntry(i, j, Double.NEGATIVE_INFINITY);
                    }
                }
            }
            
            return mask;
        }
    }
    
    /**
     * Container for attention output and weights
     */
    public static class AttentionOutput {
        public final RealMatrix output;
        public final RealMatrix attentionWeights;
        
        public AttentionOutput(RealMatrix output, RealMatrix attentionWeights) {
            this.output = output;
            this.attentionWeights = attentionWeights;
        }
    }
}

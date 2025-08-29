package com.aiprogramming.ch11;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * BERT (Bidirectional Encoder Representations from Transformers) implementation
 */
public class BERT {
    private final int dModel;
    private final int numLayers;
    private final int numHeads;
    private final int dff;
    private final int vocabSize;
    private final int maxSeqLen;
    private final int hiddenSize;
    
    // Embedding layers
    private RealMatrix tokenEmbeddings;
    private RealMatrix positionEmbeddings;
    private RealMatrix segmentEmbeddings;
    private TransformerLayer.LayerNormalization embeddingLayerNorm;
    
    // Encoder layers
    private List<TransformerLayer.EncoderLayer> encoderLayers;
    
    // Task-specific heads
    private RealMatrix mlmHead; // Masked Language Modeling head
    private RealMatrix nspHead; // Next Sentence Prediction head
    
    // Special tokens
    public static final int CLS_TOKEN = 101;
    public static final int SEP_TOKEN = 102;
    public static final int PAD_TOKEN = 0;
    public static final int MASK_TOKEN = 103;
    public static final int UNK_TOKEN = 100;
    
    public BERT(int dModel, int numLayers, int numHeads, int dff, 
                int vocabSize, int maxSeqLen) {
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dff = dff;
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.hiddenSize = dModel;
        
        initializeModel();
    }
    
    private void initializeModel() {
        // Initialize embeddings
        tokenEmbeddings = new Array2DRowRealMatrix(vocabSize, dModel);
        positionEmbeddings = new Array2DRowRealMatrix(maxSeqLen, dModel);
        segmentEmbeddings = new Array2DRowRealMatrix(2, dModel); // 2 segments max
        embeddingLayerNorm = new TransformerLayer.LayerNormalization(dModel);
        
        initializeEmbeddings();
        
        // Initialize encoder layers
        encoderLayers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            encoderLayers.add(new TransformerLayer.EncoderLayer(dModel, numHeads, dff));
        }
        
        // Initialize task-specific heads
        mlmHead = new Array2DRowRealMatrix(dModel, vocabSize);
        nspHead = new Array2DRowRealMatrix(dModel, 2); // Binary classification
        
        initializeHeads();
    }
    
    private void initializeEmbeddings() {
        Random random = new Random(42);
        double scale = Math.sqrt(2.0 / dModel);
        
        // Token embeddings
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < dModel; j++) {
                tokenEmbeddings.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
        
        // Position embeddings
        for (int i = 0; i < maxSeqLen; i++) {
            for (int j = 0; j < dModel; j++) {
                positionEmbeddings.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
        
        // Segment embeddings
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < dModel; j++) {
                segmentEmbeddings.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
    }
    
    private void initializeHeads() {
        Random random = new Random(42);
        double scale = Math.sqrt(2.0 / dModel);
        
        // MLM head
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSize; j++) {
                mlmHead.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
        
        // NSP head
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < 2; j++) {
                nspHead.setEntry(i, j, random.nextGaussian() * scale);
            }
        }
    }
    
    /**
     * Forward pass through BERT
     */
    public BERTOutput forward(RealMatrix inputIds, RealMatrix segmentIds, RealMatrix attentionMask) {
        // Embedding layer
        RealMatrix embeddings = createEmbeddings(inputIds, segmentIds);
        
        // Encoder layers
        RealMatrix hiddenStates = embeddings;
        for (TransformerLayer.EncoderLayer layer : encoderLayers) {
            hiddenStates = layer.forward(hiddenStates, attentionMask);
        }
        
        // Task-specific outputs
        RealMatrix mlmLogits = hiddenStates.multiply(mlmHead);
        RealMatrix nspLogits = hiddenStates.getSubMatrix(0, 0, 0, dModel - 1).multiply(nspHead);
        
        return new BERTOutput(hiddenStates, mlmLogits, nspLogits);
    }
    
    /**
     * Create embeddings from input tokens and segment IDs
     */
    private RealMatrix createEmbeddings(RealMatrix inputIds, RealMatrix segmentIds) {
        int batchSize = inputIds.getRowDimension();
        int seqLen = inputIds.getColumnDimension();
        RealMatrix embeddings = new Array2DRowRealMatrix(batchSize, dModel);
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                int tokenId = (int) inputIds.getEntry(i, j);
                int segmentId = (int) segmentIds.getEntry(i, j);
                
                // Token embedding
                for (int k = 0; k < dModel; k++) {
                    double tokenEmb = tokenEmbeddings.getEntry(tokenId, k);
                    double posEmb = positionEmbeddings.getEntry(j, k);
                    double segEmb = segmentEmbeddings.getEntry(segmentId, k);
                    embeddings.setEntry(i, k, tokenEmb + posEmb + segEmb);
                }
            }
        }
        
        // Apply layer normalization
        return embeddingLayerNorm.normalize(embeddings);
    }
    
    /**
     * Masked Language Modeling prediction
     */
    public RealMatrix predictMaskedTokens(RealMatrix inputIds, RealMatrix segmentIds, 
                                        RealMatrix attentionMask, List<Integer> maskedPositions) {
        BERTOutput output = forward(inputIds, segmentIds, attentionMask);
        
        // Extract predictions for masked positions
        RealMatrix maskedPredictions = new Array2DRowRealMatrix(maskedPositions.size(), vocabSize);
        
        for (int i = 0; i < maskedPositions.size(); i++) {
            int pos = maskedPositions.get(i);
            for (int j = 0; j < vocabSize; j++) {
                maskedPredictions.setEntry(i, j, output.mlmLogits.getEntry(0, j));
            }
        }
        
        return maskedPredictions;
    }
    
    /**
     * Next Sentence Prediction
     */
    public RealMatrix predictNextSentence(RealMatrix inputIds, RealMatrix segmentIds, 
                                        RealMatrix attentionMask) {
        BERTOutput output = forward(inputIds, segmentIds, attentionMask);
        return output.nspLogits;
    }
    
    /**
     * Create training data for Masked Language Modeling
     */
    public static MLMTrainingData createMLMData(RealMatrix inputIds, double maskProbability) {
        int batchSize = inputIds.getRowDimension();
        int seqLen = inputIds.getColumnDimension();
        
        RealMatrix maskedIds = inputIds.copy();
        List<Integer> maskedPositions = new ArrayList<>();
        List<Integer> originalTokens = new ArrayList<>();
        
        Random random = new Random();
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                int tokenId = (int) inputIds.getEntry(i, j);
                
                // Skip special tokens
                if (tokenId == CLS_TOKEN || tokenId == SEP_TOKEN || tokenId == PAD_TOKEN) {
                    continue;
                }
                
                if (random.nextDouble() < maskProbability) {
                    originalTokens.add(tokenId);
                    maskedPositions.add(j);
                    maskedIds.setEntry(i, j, MASK_TOKEN);
                }
            }
        }
        
        return new MLMTrainingData(maskedIds, maskedPositions, originalTokens);
    }
    
    /**
     * Create training data for Next Sentence Prediction
     */
    public static NSPTrainingData createNSPData(List<String> sentences) {
        List<RealMatrix> inputIdsList = new ArrayList<>();
        List<RealMatrix> segmentIdsList = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        Random random = new Random();
        
        for (int i = 0; i < sentences.size() - 1; i++) {
            String sentenceA = sentences.get(i);
            String sentenceB = sentences.get(i + 1);
            
            // Create positive example (consecutive sentences)
            RealMatrix inputIds = createInputIds(sentenceA, sentenceB);
            RealMatrix segmentIds = createSegmentIds(sentenceA, sentenceB);
            
            inputIdsList.add(inputIds);
            segmentIdsList.add(segmentIds);
            labels.add(1); // IsNextSentence = true
            
            // Create negative example (random sentence)
            if (i < sentences.size() - 2) {
                String randomSentence = sentences.get(random.nextInt(sentences.size()));
                inputIds = createInputIds(sentenceA, randomSentence);
                segmentIds = createSegmentIds(sentenceA, randomSentence);
                
                inputIdsList.add(inputIds);
                segmentIdsList.add(segmentIds);
                labels.add(0); // IsNextSentence = false
            }
        }
        
        return new NSPTrainingData(inputIdsList, segmentIdsList, labels);
    }
    
    /**
     * Create input IDs from two sentences
     */
    private static RealMatrix createInputIds(String sentenceA, String sentenceB) {
        // Simplified tokenization - in practice, you'd use a proper tokenizer
        String[] tokensA = sentenceA.split("\\s+");
        String[] tokensB = sentenceB.split("\\s+");
        
        List<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(CLS_TOKEN);
        
        // Add sentence A tokens
        for (String token : tokensA) {
            tokenIds.add(token.hashCode() % 1000 + 1000); // Simple hash-based tokenization
        }
        tokenIds.add(SEP_TOKEN);
        
        // Add sentence B tokens
        for (String token : tokensB) {
            tokenIds.add(token.hashCode() % 1000 + 1000);
        }
        tokenIds.add(SEP_TOKEN);
        
        // Pad to max length
        while (tokenIds.size() < 512) {
            tokenIds.add(PAD_TOKEN);
        }
        
        RealMatrix inputIds = new Array2DRowRealMatrix(1, tokenIds.size());
        for (int i = 0; i < tokenIds.size(); i++) {
            inputIds.setEntry(0, i, tokenIds.get(i));
        }
        
        return inputIds;
    }
    
    /**
     * Create segment IDs from two sentences
     */
    private static RealMatrix createSegmentIds(String sentenceA, String sentenceB) {
        String[] tokensA = sentenceA.split("\\s+");
        String[] tokensB = sentenceB.split("\\s+");
        
        List<Integer> segmentIds = new ArrayList<>();
        segmentIds.add(0); // CLS token
        
        // Sentence A tokens
        for (int i = 0; i < tokensA.length; i++) {
            segmentIds.add(0);
        }
        segmentIds.add(0); // SEP token
        
        // Sentence B tokens
        for (int i = 0; i < tokensB.length; i++) {
            segmentIds.add(1);
        }
        segmentIds.add(1); // SEP token
        
        // Pad
        while (segmentIds.size() < 512) {
            segmentIds.add(0);
        }
        
        RealMatrix segmentMatrix = new Array2DRowRealMatrix(1, segmentIds.size());
        for (int i = 0; i < segmentIds.size(); i++) {
            segmentMatrix.setEntry(0, i, segmentIds.get(i));
        }
        
        return segmentMatrix;
    }
    
    /**
     * Container for BERT output
     */
    public static class BERTOutput {
        public final RealMatrix hiddenStates;
        public final RealMatrix mlmLogits;
        public final RealMatrix nspLogits;
        
        public BERTOutput(RealMatrix hiddenStates, RealMatrix mlmLogits, RealMatrix nspLogits) {
            this.hiddenStates = hiddenStates;
            this.mlmLogits = mlmLogits;
            this.nspLogits = nspLogits;
        }
    }
    
    /**
     * Container for MLM training data
     */
    public static class MLMTrainingData {
        public final RealMatrix maskedIds;
        public final List<Integer> maskedPositions;
        public final List<Integer> originalTokens;
        
        public MLMTrainingData(RealMatrix maskedIds, List<Integer> maskedPositions, 
                             List<Integer> originalTokens) {
            this.maskedIds = maskedIds;
            this.maskedPositions = maskedPositions;
            this.originalTokens = originalTokens;
        }
    }
    
    /**
     * Container for NSP training data
     */
    public static class NSPTrainingData {
        public final List<RealMatrix> inputIds;
        public final List<RealMatrix> segmentIds;
        public final List<Integer> labels;
        
        public NSPTrainingData(List<RealMatrix> inputIds, List<RealMatrix> segmentIds, 
                             List<Integer> labels) {
            this.inputIds = inputIds;
            this.segmentIds = segmentIds;
            this.labels = labels;
        }
    }
}

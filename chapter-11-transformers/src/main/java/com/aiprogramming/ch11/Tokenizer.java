package com.aiprogramming.ch11;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Simple tokenizer implementation for transformer models
 */
public class Tokenizer {
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final int vocabSize;
    private final int maxLength;
    
    // Special tokens
    public static final String PAD_TOKEN = "[PAD]";
    public static final String UNK_TOKEN = "[UNK]";
    public static final String CLS_TOKEN = "[CLS]";
    public static final String SEP_TOKEN = "[SEP]";
    public static final String MASK_TOKEN = "[MASK]";
    
    public Tokenizer(int vocabSize, int maxLength) {
        this.vocabSize = vocabSize;
        this.maxLength = maxLength;
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        
        initializeVocabulary();
    }
    
    private void initializeVocabulary() {
        // Add special tokens
        vocab.put(PAD_TOKEN, 0);
        vocab.put(UNK_TOKEN, 100);
        vocab.put(CLS_TOKEN, 101);
        vocab.put(SEP_TOKEN, 102);
        vocab.put(MASK_TOKEN, 103);
        
        reverseVocab.put(0, PAD_TOKEN);
        reverseVocab.put(100, UNK_TOKEN);
        reverseVocab.put(101, CLS_TOKEN);
        reverseVocab.put(102, SEP_TOKEN);
        reverseVocab.put(103, MASK_TOKEN);
        
        // Add common words and subwords
        int tokenId = 104;
        String[] commonWords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
            "itself", "ourselves", "yourselves", "themselves"
        };
        
        for (String word : commonWords) {
            if (tokenId < vocabSize) {
                vocab.put(word, tokenId);
                reverseVocab.put(tokenId, word);
                tokenId++;
            }
        }
        
        // Add common punctuation
        String[] punctuation = {".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "\"", "'"};
        for (String punct : punctuation) {
            if (tokenId < vocabSize) {
                vocab.put(punct, tokenId);
                reverseVocab.put(tokenId, punct);
                tokenId++;
            }
        }
    }
    
    /**
     * Tokenize a text into tokens
     */
    public List<String> tokenize(String text) {
        // Convert to lowercase and normalize whitespace
        text = text.toLowerCase().trim();
        text = text.replaceAll("\\s+", " ");
        
        // Split into words
        String[] words = text.split("\\s+");
        List<String> tokens = new ArrayList<>();
        
        for (String word : words) {
            // Handle punctuation
            List<String> wordTokens = splitWord(word);
            tokens.addAll(wordTokens);
        }
        
        return tokens;
    }
    
    /**
     * Split a word into subwords/tokens
     */
    private List<String> splitWord(String word) {
        List<String> tokens = new ArrayList<>();
        
        // Remove punctuation from the end
        String cleanWord = word.replaceAll("[^a-zA-Z0-9]", "");
        
        if (cleanWord.isEmpty()) {
            // Handle pure punctuation
            String punct = word.replaceAll("[a-zA-Z0-9]", "");
            if (!punct.isEmpty()) {
                tokens.add(punct);
            }
            return tokens;
        }
        
        // Check if word is in vocabulary
        if (vocab.containsKey(cleanWord)) {
            tokens.add(cleanWord);
        } else {
            // Apply subword tokenization (simplified)
            tokens.addAll(splitIntoSubwords(cleanWord));
        }
        
        // Add trailing punctuation
        String trailingPunct = word.replaceAll("[a-zA-Z0-9]", "");
        if (!trailingPunct.isEmpty()) {
            tokens.add(trailingPunct);
        }
        
        return tokens;
    }
    
    /**
     * Split word into subwords using simple character-level approach
     */
    private List<String> splitIntoSubwords(String word) {
        List<String> subwords = new ArrayList<>();
        
        if (word.length() <= 3) {
            // Short words become unknown tokens
            subwords.add(UNK_TOKEN);
        } else {
            // Split into character n-grams
            for (int i = 0; i < word.length() - 2; i++) {
                String subword = word.substring(i, i + 3);
                if (vocab.containsKey(subword)) {
                    subwords.add(subword);
                } else {
                    subwords.add(UNK_TOKEN);
                }
            }
        }
        
        return subwords;
    }
    
    /**
     * Convert tokens to token IDs
     */
    public List<Integer> encode(List<String> tokens) {
        List<Integer> tokenIds = new ArrayList<>();
        
        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                tokenIds.add(vocab.get(token));
            } else {
                tokenIds.add(vocab.get(UNK_TOKEN));
            }
        }
        
        return tokenIds;
    }
    
    /**
     * Convert token IDs back to tokens
     */
    public List<String> decode(List<Integer> tokenIds) {
        List<String> tokens = new ArrayList<>();
        
        for (Integer tokenId : tokenIds) {
            if (reverseVocab.containsKey(tokenId)) {
                tokens.add(reverseVocab.get(tokenId));
            } else {
                tokens.add(UNK_TOKEN);
            }
        }
        
        return tokens;
    }
    
    /**
     * Encode text with special tokens for BERT
     */
    public List<Integer> encodeBERT(String text) {
        List<String> tokens = tokenize(text);
        List<Integer> tokenIds = new ArrayList<>();
        
        // Add CLS token
        tokenIds.add(vocab.get(CLS_TOKEN));
        
        // Add text tokens
        tokenIds.addAll(encode(tokens));
        
        // Add SEP token
        tokenIds.add(vocab.get(SEP_TOKEN));
        
        // Pad to max length
        while (tokenIds.size() < maxLength) {
            tokenIds.add(vocab.get(PAD_TOKEN));
        }
        
        // Truncate if too long
        if (tokenIds.size() > maxLength) {
            tokenIds = tokenIds.subList(0, maxLength);
        }
        
        return tokenIds;
    }
    
    /**
     * Encode sentence pair for BERT
     */
    public List<Integer> encodeSentencePair(String sentenceA, String sentenceB) {
        List<String> tokensA = tokenize(sentenceA);
        List<String> tokensB = tokenize(sentenceB);
        List<Integer> tokenIds = new ArrayList<>();
        
        // Add CLS token
        tokenIds.add(vocab.get(CLS_TOKEN));
        
        // Add sentence A tokens
        tokenIds.addAll(encode(tokensA));
        
        // Add SEP token
        tokenIds.add(vocab.get(SEP_TOKEN));
        
        // Add sentence B tokens
        tokenIds.addAll(encode(tokensB));
        
        // Add SEP token
        tokenIds.add(vocab.get(SEP_TOKEN));
        
        // Pad to max length
        while (tokenIds.size() < maxLength) {
            tokenIds.add(vocab.get(PAD_TOKEN));
        }
        
        // Truncate if too long
        if (tokenIds.size() > maxLength) {
            tokenIds = tokenIds.subList(0, maxLength);
        }
        
        return tokenIds;
    }
    
    /**
     * Create segment IDs for sentence pair
     */
    public List<Integer> createSegmentIds(String sentenceA, String sentenceB) {
        List<String> tokensA = tokenize(sentenceA);
        List<String> tokensB = tokenize(sentenceB);
        List<Integer> segmentIds = new ArrayList<>();
        
        // CLS token
        segmentIds.add(0);
        
        // Sentence A tokens
        for (int i = 0; i < tokensA.size(); i++) {
            segmentIds.add(0);
        }
        
        // SEP token
        segmentIds.add(0);
        
        // Sentence B tokens
        for (int i = 0; i < tokensB.size(); i++) {
            segmentIds.add(1);
        }
        
        // SEP token
        segmentIds.add(1);
        
        // Pad
        while (segmentIds.size() < maxLength) {
            segmentIds.add(0);
        }
        
        // Truncate if too long
        if (segmentIds.size() > maxLength) {
            segmentIds = segmentIds.subList(0, maxLength);
        }
        
        return segmentIds;
    }
    
    /**
     * Create attention mask
     */
    public List<Integer> createAttentionMask(List<Integer> tokenIds) {
        List<Integer> attentionMask = new ArrayList<>();
        
        for (Integer tokenId : tokenIds) {
            if (tokenId == vocab.get(PAD_TOKEN)) {
                attentionMask.add(0);
            } else {
                attentionMask.add(1);
            }
        }
        
        return attentionMask;
    }
    
    /**
     * Get vocabulary size
     */
    public int getVocabSize() {
        return vocabSize;
    }
    
    /**
     * Get maximum sequence length
     */
    public int getMaxLength() {
        return maxLength;
    }
    
    /**
     * Get vocabulary
     */
    public Map<String, Integer> getVocab() {
        return new HashMap<>(vocab);
    }
}

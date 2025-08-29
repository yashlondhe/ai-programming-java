package com.aiprogramming.ch09.utils;

import java.util.*;

/**
 * Vocabulary management for text processing.
 * 
 * This class handles character-to-index mapping and one-hot encoding
 * for text generation and sentiment analysis tasks.
 */
public class Vocabulary {
    private Map<String, Integer> charToIndex;
    private Map<Integer, String> indexToChar;
    private Set<String> uniqueChars;
    private boolean isBuilt;
    
    /**
     * Constructor for Vocabulary.
     */
    public Vocabulary() {
        this.charToIndex = new HashMap<>();
        this.indexToChar = new HashMap<>();
        this.uniqueChars = new HashSet<>();
        this.isBuilt = false;
    }
    
    /**
     * Build vocabulary from text.
     * 
     * @param text Input text
     */
    public void buildFromText(String text) {
        uniqueChars.clear();
        charToIndex.clear();
        indexToChar.clear();
        
        // Extract unique characters
        String[] characters = text.split("");
        for (String ch : characters) {
            uniqueChars.add(ch);
        }
        
        // Create mappings
        int index = 0;
        for (String ch : uniqueChars) {
            charToIndex.put(ch, index);
            indexToChar.put(index, ch);
            index++;
        }
        
        isBuilt = true;
    }
    
    /**
     * Convert character to one-hot encoded vector.
     * 
     * @param character Input character
     * @return One-hot encoded vector
     */
    public double[] oneHotEncode(String character) {
        if (!isBuilt) {
            throw new IllegalStateException("Vocabulary must be built before encoding");
        }
        
        if (!charToIndex.containsKey(character)) {
            throw new IllegalArgumentException("Character not in vocabulary: " + character);
        }
        
        int index = charToIndex.get(character);
        double[] encoding = new double[uniqueChars.size()];
        encoding[index] = 1.0;
        
        return encoding;
    }
    
    /**
     * Convert one-hot encoded vector to character.
     * 
     * @param encoding One-hot encoded vector
     * @return Character
     */
    public String oneHotDecode(double[] encoding) {
        if (!isBuilt) {
            throw new IllegalStateException("Vocabulary must be built before decoding");
        }
        
        int maxIndex = 0;
        double maxValue = encoding[0];
        
        for (int i = 1; i < encoding.length; i++) {
            if (encoding[i] > maxValue) {
                maxValue = encoding[i];
                maxIndex = i;
            }
        }
        
        return indexToChar.get(maxIndex);
    }
    
    /**
     * Get character by index.
     * 
     * @param index Character index
     * @return Character
     */
    public String getCharacter(int index) {
        if (!isBuilt) {
            throw new IllegalStateException("Vocabulary must be built before accessing");
        }
        
        return indexToChar.get(index);
    }
    
    /**
     * Get index by character.
     * 
     * @param character Character
     * @return Character index
     */
    public int getIndex(String character) {
        if (!isBuilt) {
            throw new IllegalStateException("Vocabulary must be built before accessing");
        }
        
        return charToIndex.getOrDefault(character, -1);
    }
    
    /**
     * Check if vocabulary contains a character.
     * 
     * @param character Character to check
     * @return True if character is in vocabulary
     */
    public boolean contains(String character) {
        return charToIndex.containsKey(character);
    }
    
    /**
     * Get vocabulary size.
     * 
     * @return Number of unique characters
     */
    public int getSize() {
        return uniqueChars.size();
    }
    
    /**
     * Get all characters in vocabulary.
     * 
     * @return Set of unique characters
     */
    public Set<String> getCharacters() {
        return new HashSet<>(uniqueChars);
    }
    
    /**
     * Get character-to-index mapping.
     * 
     * @return Map of character to index
     */
    public Map<String, Integer> getCharToIndex() {
        return new HashMap<>(charToIndex);
    }
    
    /**
     * Get index-to-character mapping.
     * 
     * @return Map of index to character
     */
    public Map<Integer, String> getIndexToChar() {
        return new HashMap<>(indexToChar);
    }
    
    /**
     * Check if vocabulary is built.
     * 
     * @return True if vocabulary has been built
     */
    public boolean isBuilt() {
        return isBuilt;
    }
    
    /**
     * Clear vocabulary.
     */
    public void clear() {
        charToIndex.clear();
        indexToChar.clear();
        uniqueChars.clear();
        isBuilt = false;
    }
    
    /**
     * Get string representation of vocabulary.
     * 
     * @return String representation
     */
    @Override
    public String toString() {
        if (!isBuilt) {
            return "Vocabulary (not built)";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("Vocabulary (size: ").append(getSize()).append("):\n");
        
        for (int i = 0; i < getSize(); i++) {
            String ch = getCharacter(i);
            sb.append("  ").append(i).append(" -> '").append(ch).append("'\n");
        }
        
        return sb.toString();
    }
}

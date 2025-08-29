package com.aiprogramming.ch10;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Named Entity Recognition (NER) implementation
 * Uses pattern matching and rule-based approaches to identify entities
 */
public class NamedEntityRecognizer {
    
    private final TextPreprocessor preprocessor;
    private final Map<String, List<String>> entityPatterns;
    private final Map<String, Pattern> compiledPatterns;
    
    // Entity types
    public static final String PERSON = "PERSON";
    public static final String ORGANIZATION = "ORGANIZATION";
    public static final String LOCATION = "LOCATION";
    public static final String DATE = "DATE";
    public static final String MONEY = "MONEY";
    public static final String PERCENT = "PERCENT";
    public static final String TIME = "TIME";
    
    public NamedEntityRecognizer() {
        this.preprocessor = new TextPreprocessor();
        this.entityPatterns = new HashMap<>();
        this.compiledPatterns = new HashMap<>();
        initializePatterns();
    }
    
    /**
     * Initialize entity recognition patterns
     */
    private void initializePatterns() {
        // Person patterns (simplified)
        List<String> personPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+ [A-Z][a-z]+\\b",  // First Last
            "\\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\\b",  // First Middle Last
            "\\b[A-Z][a-z]+ [A-Z]\\.[A-Z][a-z]+\\b"  // First M. Last
        );
        entityPatterns.put(PERSON, personPatterns);
        
        // Organization patterns
        List<String> orgPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation|Organization)\\b",
            "\\b[A-Z][a-z]+ [A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation|Organization)\\b",
            "\\b[A-Z]+\\b"  // All caps words (like IBM, NASA)
        );
        entityPatterns.put(ORGANIZATION, orgPatterns);
        
        // Location patterns
        List<String> locationPatterns = Arrays.asList(
            "\\b[A-Z][a-z]+, [A-Z]{2}\\b",  // City, State
            "\\b[A-Z][a-z]+ [A-Z][a-z]+, [A-Z]{2}\\b",  // City Name, State
            "\\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard|Drive|Lane)\\b",
            "\\b[A-Z][a-z]+ (City|Town|Village|County|State|Country)\\b"
        );
        entityPatterns.put(LOCATION, locationPatterns);
        
        // Date patterns
        List<String> datePatterns = Arrays.asList(
            "\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b",  // MM/DD/YYYY
            "\\b\\d{4}-\\d{1,2}-\\d{1,2}\\b",  // YYYY-MM-DD
            "\\b(January|February|March|April|May|June|July|August|September|October|November|December) \\d{1,2},? \\d{4}\\b",
            "\\b\\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \\d{4}\\b"
        );
        entityPatterns.put(DATE, datePatterns);
        
        // Money patterns
        List<String> moneyPatterns = Arrays.asList(
            "\\$\\d+(\\.\\d{2})?\\b",
            "\\b\\d+(\\.\\d{2})? (dollars|USD)\\b",
            "\\b\\d+ (cents|pennies)\\b"
        );
        entityPatterns.put(MONEY, moneyPatterns);
        
        // Percentage patterns
        List<String> percentPatterns = Arrays.asList(
            "\\b\\d+(\\.\\d+)?%\\b",
            "\\b\\d+(\\.\\d+)? percent\\b"
        );
        entityPatterns.put(PERCENT, percentPatterns);
        
        // Time patterns
        List<String> timePatterns = Arrays.asList(
            "\\b\\d{1,2}:\\d{2} (AM|PM)\\b",
            "\\b\\d{1,2}:\\d{2}\\b",
            "\\b(noon|midnight)\\b"
        );
        entityPatterns.put(TIME, timePatterns);
        
        // Compile all patterns
        for (Map.Entry<String, List<String>> entry : entityPatterns.entrySet()) {
            for (String pattern : entry.getValue()) {
                compiledPatterns.put(pattern, Pattern.compile(pattern));
            }
        }
    }
    
    /**
     * Extract named entities from text
     * @param text Input text
     * @return Map of entity type to list of entities found
     */
    public Map<String, List<String>> extractEntities(String text) {
        Map<String, List<String>> entities = new HashMap<>();
        
        // Initialize entity lists
        for (String entityType : entityPatterns.keySet()) {
            entities.put(entityType, new ArrayList<>());
        }
        
        // Extract entities for each type
        for (Map.Entry<String, List<String>> entry : entityPatterns.entrySet()) {
            String entityType = entry.getKey();
            List<String> patterns = entry.getValue();
            
            for (String pattern : patterns) {
                Pattern compiledPattern = compiledPatterns.get(pattern);
                java.util.regex.Matcher matcher = compiledPattern.matcher(text);
                
                while (matcher.find()) {
                    String entity = matcher.group();
                    if (!entities.get(entityType).contains(entity)) {
                        entities.get(entityType).add(entity);
                    }
                }
            }
        }
        
        return entities;
    }
    
    /**
     * Extract entities with their positions in the text
     * @param text Input text
     * @return List of entity objects with type, text, and position
     */
    public List<Entity> extractEntitiesWithPositions(String text) {
        List<Entity> entities = new ArrayList<>();
        
        for (Map.Entry<String, List<String>> entry : entityPatterns.entrySet()) {
            String entityType = entry.getKey();
            List<String> patterns = entry.getValue();
            
            for (String pattern : patterns) {
                Pattern compiledPattern = compiledPatterns.get(pattern);
                java.util.regex.Matcher matcher = compiledPattern.matcher(text);
                
                while (matcher.find()) {
                    String entityText = matcher.group();
                    int start = matcher.start();
                    int end = matcher.end();
                    
                    Entity entity = new Entity(entityType, entityText, start, end);
                    entities.add(entity);
                }
            }
        }
        
        // Sort by position
        entities.sort(Comparator.comparingInt(Entity::getStart));
        
        return entities;
    }
    
    /**
     * Check if a word is likely a person name
     * @param word Word to check
     * @return true if likely a person name
     */
    public boolean isPersonName(String word) {
        // Simple heuristics for person names
        if (word == null || word.length() < 2) {
            return false;
        }
        
        // Check if it matches person patterns
        for (String pattern : entityPatterns.get(PERSON)) {
            Pattern compiledPattern = compiledPatterns.get(pattern);
            if (compiledPattern.matcher(word).matches()) {
                return true;
            }
        }
        
        // Additional heuristics
        String[] parts = word.split("\\s+");
        if (parts.length >= 2) {
            // Check if each part starts with capital letter
            for (String part : parts) {
                if (part.length() > 0 && !Character.isUpperCase(part.charAt(0))) {
                    return false;
                }
            }
            return true;
        }
        
        return false;
    }
    
    /**
     * Check if a word is likely an organization name
     * @param word Word to check
     * @return true if likely an organization name
     */
    public boolean isOrganizationName(String word) {
        if (word == null || word.length() < 2) {
            return false;
        }
        
        // Check if it matches organization patterns
        for (String pattern : entityPatterns.get(ORGANIZATION)) {
            Pattern compiledPattern = compiledPatterns.get(pattern);
            if (compiledPattern.matcher(word).matches()) {
                return true;
            }
        }
        
        // Check if it's all caps (like IBM, NASA)
        if (word.equals(word.toUpperCase()) && word.length() > 1) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Print extracted entities
     * @param text Input text
     */
    public void printEntities(String text) {
        Map<String, List<String>> entities = extractEntities(text);
        
        System.out.println("Extracted Entities:");
        System.out.println("Text: " + text);
        System.out.println();
        
        for (Map.Entry<String, List<String>> entry : entities.entrySet()) {
            String entityType = entry.getKey();
            List<String> entityList = entry.getValue();
            
            if (!entityList.isEmpty()) {
                System.out.println(entityType + ":");
                for (String entity : entityList) {
                    System.out.println("  - " + entity);
                }
                System.out.println();
            }
        }
    }
    
    /**
     * Entity class to hold entity information
     */
    public static class Entity {
        private final String type;
        private final String text;
        private final int start;
        private final int end;
        
        public Entity(String type, String text, int start, int end) {
            this.type = type;
            this.text = text;
            this.start = start;
            this.end = end;
        }
        
        public String getType() { return type; }
        public String getText() { return text; }
        public int getStart() { return start; }
        public int getEnd() { return end; }
        
        @Override
        public String toString() {
            return String.format("%s('%s', %d-%d)", type, text, start, end);
        }
    }
}

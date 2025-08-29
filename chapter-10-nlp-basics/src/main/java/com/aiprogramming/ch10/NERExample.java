package com.aiprogramming.ch10;

import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Example demonstrating Named Entity Recognition (NER) functionality
 */
public class NERExample {
    
    public static void main(String[] args) {
        System.out.println("=== Named Entity Recognition Example ===\n");
        
        NamedEntityRecognizer ner = new NamedEntityRecognizer();
        
        // Sample texts with various entities
        List<String> sampleTexts = Arrays.asList(
            "John Smith works at Microsoft Corporation in Seattle, WA.",
            "Apple Inc. released the iPhone 15 on September 12, 2023.",
            "The meeting is scheduled for 2:30 PM on January 15, 2024.",
            "IBM and Google are competing in artificial intelligence research.",
            "Sarah Johnson lives at 123 Main Street, New York City, NY.",
            "The project cost $50,000 and took 6 months to complete.",
            "NASA launched the Mars rover on July 30, 2020.",
            "Dr. Michael Brown teaches at Stanford University in California.",
            "The stock price increased by 15% this quarter.",
            "Amazon.com Inc. is headquartered in Seattle, Washington."
        );
        
        System.out.println("Sample texts for entity extraction:");
        for (int i = 0; i < sampleTexts.size(); i++) {
            System.out.println((i + 1) + ". " + sampleTexts.get(i));
        }
        System.out.println();
        
        // Extract entities from each text
        System.out.println("=== Entity Extraction Results ===");
        for (int i = 0; i < sampleTexts.size(); i++) {
            String text = sampleTexts.get(i);
            System.out.println("Text " + (i + 1) + ": " + text);
            
            Map<String, List<String>> entities = ner.extractEntities(text);
            
            boolean foundEntities = false;
            for (Map.Entry<String, List<String>> entry : entities.entrySet()) {
                String entityType = entry.getKey();
                List<String> entityList = entry.getValue();
                
                if (!entityList.isEmpty()) {
                    foundEntities = true;
                    System.out.println("  " + entityType + ":");
                    for (String entity : entityList) {
                        System.out.println("    - " + entity);
                    }
                }
            }
            
            if (!foundEntities) {
                System.out.println("  No entities found.");
            }
            System.out.println();
        }
        
        // Extract entities with positions
        System.out.println("=== Entity Extraction with Positions ===");
        String complexText = "Dr. Sarah Johnson from Stanford University in California " +
                           "will present at the conference on March 15, 2024 at 3:00 PM. " +
                           "The event costs $500 and will be held at 456 Tech Avenue, San Francisco, CA.";
        
        System.out.println("Complex text: " + complexText);
        System.out.println();
        
        List<NamedEntityRecognizer.Entity> entitiesWithPositions = ner.extractEntitiesWithPositions(complexText);
        
        System.out.println("Entities with positions:");
        for (NamedEntityRecognizer.Entity entity : entitiesWithPositions) {
            System.out.printf("  %s: '%s' (position %d-%d)%n", 
                entity.getType(), entity.getText(), entity.getStart(), entity.getEnd());
        }
        System.out.println();
        
        // Test entity type detection
        System.out.println("=== Entity Type Detection ===");
        List<String> testWords = Arrays.asList(
            "John Smith",
            "Microsoft Corporation",
            "Seattle, WA",
            "September 12, 2023",
            "$50,000",
            "15%",
            "2:30 PM",
            "IBM",
            "NASA",
            "Dr. Michael Brown",
            "Stanford University",
            "California"
        );
        
        System.out.println("Entity type detection:");
        for (String word : testWords) {
            boolean isPerson = ner.isPersonName(word);
            boolean isOrg = ner.isOrganizationName(word);
            
            System.out.printf("  '%s': ", word);
            if (isPerson) {
                System.out.print("PERSON ");
            }
            if (isOrg) {
                System.out.print("ORGANIZATION ");
            }
            if (!isPerson && !isOrg) {
                System.out.print("OTHER ");
            }
            System.out.println();
        }
        System.out.println();
        
        // Interactive entity extraction
        System.out.println("=== Interactive Entity Extraction ===");
        List<String> interactiveTexts = Arrays.asList(
            "The CEO of Apple Inc., Tim Cook, announced new products on January 27, 2024.",
            "Google's headquarters in Mountain View, CA employs over 100,000 people.",
            "The movie Titanic was released on December 19, 1997 and grossed $2.2 billion.",
            "NASA's Perseverance rover landed on Mars on February 18, 2021 at 3:55 PM EST.",
            "Amazon.com Inc. was founded by Jeff Bezos in Seattle, Washington in 1994."
        );
        
        for (String text : interactiveTexts) {
            ner.printEntities(text);
        }
        
        // Entity statistics
        System.out.println("=== Entity Statistics ===");
        Map<String, Integer> entityTypeCounts = new HashMap<>();
        
        for (String text : sampleTexts) {
            Map<String, List<String>> entities = ner.extractEntities(text);
            for (Map.Entry<String, List<String>> entry : entities.entrySet()) {
                String entityType = entry.getKey();
                int count = entry.getValue().size();
                entityTypeCounts.put(entityType, entityTypeCounts.getOrDefault(entityType, 0) + count);
            }
        }
        
        System.out.println("Entity type distribution across all texts:");
        for (Map.Entry<String, Integer> entry : entityTypeCounts.entrySet()) {
            System.out.printf("  %s: %d entities%n", entry.getKey(), entry.getValue());
        }
        System.out.println();
        
        // Pattern matching examples
        System.out.println("=== Pattern Matching Examples ===");
        List<String> patternExamples = Arrays.asList(
            "Date formats: 12/25/2023, 2023-12-25, December 25, 2023",
            "Money formats: $100, $1,500.50, 500 dollars, 25 cents",
            "Time formats: 9:30 AM, 14:30, noon, midnight",
            "Percentage formats: 25%, 12.5 percent",
            "Address formats: 123 Main Street, New York City, NY"
        );
        
        for (String example : patternExamples) {
            System.out.println("Text: " + example);
            Map<String, List<String>> entities = ner.extractEntities(example);
            
            for (Map.Entry<String, List<String>> entry : entities.entrySet()) {
                String entityType = entry.getKey();
                List<String> entityList = entry.getValue();
                
                if (!entityList.isEmpty()) {
                    System.out.println("  " + entityType + ": " + entityList);
                }
            }
            System.out.println();
        }
        
        System.out.println("=== Named Entity Recognition Example Complete ===");
    }
}

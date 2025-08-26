package com.aiprogramming.ch06;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Apriori algorithm for association rule learning
 */
public class Apriori {
    private final double minSupport;
    private final double minConfidence;
    private List<Set<String>> frequentItemsets;
    private List<AssociationRule> associationRules;
    private boolean trained;
    
    public Apriori(double minSupport, double minConfidence) {
        if (minSupport <= 0 || minSupport > 1) {
            throw new IllegalArgumentException("MinSupport must be between 0 and 1");
        }
        if (minConfidence <= 0 || minConfidence > 1) {
            throw new IllegalArgumentException("MinConfidence must be between 0 and 1");
        }
        
        this.minSupport = minSupport;
        this.minConfidence = minConfidence;
        this.frequentItemsets = new ArrayList<>();
        this.associationRules = new ArrayList<>();
        this.trained = false;
    }
    
    /**
     * Train the Apriori model
     */
    public void fit(List<Set<String>> transactions) {
        if (transactions.isEmpty()) {
            throw new IllegalArgumentException("Transactions cannot be empty");
        }
        
        frequentItemsets.clear();
        associationRules.clear();
        
        // Generate frequent 1-itemsets
        Map<Set<String>, Integer> frequent1Itemsets = generateFrequent1Itemsets(transactions);
        frequentItemsets.addAll(frequent1Itemsets.keySet());
        
        // Generate frequent k-itemsets for k > 1
        Map<Set<String>, Integer> currentFrequentItemsets = frequent1Itemsets;
        int k = 2;
        
        while (!currentFrequentItemsets.isEmpty()) {
            // Generate candidate k-itemsets
            Set<Set<String>> candidates = generateCandidates(currentFrequentItemsets.keySet(), k);
            
            // Count support for candidates
            Map<Set<String>, Integer> candidateCounts = countSupport(candidates, transactions);
            
            // Filter by minimum support
            currentFrequentItemsets = candidateCounts.entrySet().stream()
                    .filter(entry -> (double) entry.getValue() / transactions.size() >= minSupport)
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            
            frequentItemsets.addAll(currentFrequentItemsets.keySet());
            k++;
        }
        
        // Generate association rules
        generateAssociationRules(transactions);
        
        this.trained = true;
    }
    
    /**
     * Generate frequent 1-itemsets
     */
    private Map<Set<String>, Integer> generateFrequent1Itemsets(List<Set<String>> transactions) {
        Map<String, Integer> itemCounts = new HashMap<>();
        
        // Count individual items
        for (Set<String> transaction : transactions) {
            for (String item : transaction) {
                itemCounts.put(item, itemCounts.getOrDefault(item, 0) + 1);
            }
        }
        
        // Filter by minimum support
        return itemCounts.entrySet().stream()
                .filter(entry -> (double) entry.getValue() / transactions.size() >= minSupport)
                .collect(Collectors.toMap(
                    entry -> Set.of(entry.getKey()),
                    Map.Entry::getValue
                ));
    }
    
    /**
     * Generate candidate k-itemsets from frequent (k-1)-itemsets
     */
    private Set<Set<String>> generateCandidates(Set<Set<String>> frequentItemsets, int k) {
        Set<Set<String>> candidates = new HashSet<>();
        
        List<Set<String>> itemsetsList = new ArrayList<>(frequentItemsets);
        
        for (int i = 0; i < itemsetsList.size(); i++) {
            for (int j = i + 1; j < itemsetsList.size(); j++) {
                Set<String> itemset1 = itemsetsList.get(i);
                Set<String> itemset2 = itemsetsList.get(j);
                
                // Check if itemsets can be joined
                if (canJoin(itemset1, itemset2, k)) {
                    Set<String> candidate = new HashSet<>(itemset1);
                    candidate.addAll(itemset2);
                    candidates.add(candidate);
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Check if two itemsets can be joined to form a k-itemset
     */
    private boolean canJoin(Set<String> itemset1, Set<String> itemset2, int k) {
        if (itemset1.size() != k - 1 || itemset2.size() != k - 1) {
            return false;
        }
        
        // Check if first k-2 elements are the same
        List<String> list1 = new ArrayList<>(itemset1);
        List<String> list2 = new ArrayList<>(itemset2);
        Collections.sort(list1);
        Collections.sort(list2);
        
        for (int i = 0; i < k - 2; i++) {
            if (!list1.get(i).equals(list2.get(i))) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Count support for candidate itemsets
     */
    private Map<Set<String>, Integer> countSupport(Set<Set<String>> candidates, List<Set<String>> transactions) {
        Map<Set<String>, Integer> counts = new HashMap<>();
        
        for (Set<String> candidate : candidates) {
            counts.put(candidate, 0);
        }
        
        for (Set<String> transaction : transactions) {
            for (Set<String> candidate : candidates) {
                if (transaction.containsAll(candidate)) {
                    counts.put(candidate, counts.get(candidate) + 1);
                }
            }
        }
        
        return counts;
    }
    
    /**
     * Generate association rules from frequent itemsets
     */
    private void generateAssociationRules(List<Set<String>> transactions) {
        for (Set<String> itemset : frequentItemsets) {
            if (itemset.size() < 2) {
                continue;
            }
            
            // Generate all possible rules from this itemset
            List<Set<String>> subsets = generateSubsets(itemset);
            
            for (Set<String> antecedent : subsets) {
                if (antecedent.isEmpty() || antecedent.size() == itemset.size()) {
                    continue;
                }
                
                Set<String> consequent = new HashSet<>(itemset);
                consequent.removeAll(antecedent);
                
                double confidence = calculateConfidence(antecedent, itemset, transactions);
                
                if (confidence >= minConfidence) {
                    associationRules.add(new AssociationRule(antecedent, consequent, confidence));
                }
            }
        }
    }
    
    /**
     * Generate all subsets of an itemset
     */
    private List<Set<String>> generateSubsets(Set<String> itemset) {
        List<Set<String>> subsets = new ArrayList<>();
        List<String> items = new ArrayList<>(itemset);
        int n = items.size();
        
        // Generate all possible subsets using bit manipulation
        for (int i = 0; i < (1 << n); i++) {
            Set<String> subset = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) {
                    subset.add(items.get(j));
                }
            }
            subsets.add(subset);
        }
        
        return subsets;
    }
    
    /**
     * Calculate confidence of a rule
     */
    private double calculateConfidence(Set<String> antecedent, Set<String> itemset, List<Set<String>> transactions) {
        int antecedentCount = 0;
        int itemsetCount = 0;
        
        for (Set<String> transaction : transactions) {
            if (transaction.containsAll(antecedent)) {
                antecedentCount++;
            }
            if (transaction.containsAll(itemset)) {
                itemsetCount++;
            }
        }
        
        return antecedentCount > 0 ? (double) itemsetCount / antecedentCount : 0.0;
    }
    
    /**
     * Get frequent itemsets
     */
    public List<Set<String>> getFrequentItemsets() {
        return new ArrayList<>(frequentItemsets);
    }
    
    /**
     * Get association rules
     */
    public List<AssociationRule> getAssociationRules() {
        return new ArrayList<>(associationRules);
    }
    
    /**
     * Get the name of the algorithm
     */
    public String getName() {
        return "Apriori";
    }
    
    /**
     * Inner class to represent association rules
     */
    public static class AssociationRule {
        private final Set<String> antecedent;
        private final Set<String> consequent;
        private final double confidence;
        
        public AssociationRule(Set<String> antecedent, Set<String> consequent, double confidence) {
            this.antecedent = new HashSet<>(antecedent);
            this.consequent = new HashSet<>(consequent);
            this.confidence = confidence;
        }
        
        public Set<String> getAntecedent() {
            return new HashSet<>(antecedent);
        }
        
        public Set<String> getConsequent() {
            return new HashSet<>(consequent);
        }
        
        public double getConfidence() {
            return confidence;
        }
        
        @Override
        public String toString() {
            return String.format("%s -> %s (confidence: %.3f)", antecedent, consequent, confidence);
        }
    }
}

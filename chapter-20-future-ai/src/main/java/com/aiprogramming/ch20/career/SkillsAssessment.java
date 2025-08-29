package com.aiprogramming.ch20.career;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * AI Skills Assessment Framework
 * 
 * Provides comprehensive assessment of AI-related skills and competencies,
 * helping individuals understand their current skill level and identify
 * areas for improvement in their AI career development.
 */
public class SkillsAssessment {
    
    private static final Logger logger = LoggerFactory.getLogger(SkillsAssessment.class);
    
    private final Map<String, SkillCategory> skillCategories;
    private final List<AssessmentResult> assessmentHistory;
    
    public SkillsAssessment() {
        this.skillCategories = new HashMap<>();
        this.assessmentHistory = new ArrayList<>();
        initializeSkillCategories();
    }
    
    /**
     * Represents a skill category with multiple skills
     */
    public static class SkillCategory {
        private final String name;
        private final String description;
        private final List<Skill> skills;
        private final double weight; // Importance weight for overall assessment
        
        public SkillCategory(String name, String description, double weight) {
            this.name = name;
            this.description = description;
            this.skills = new ArrayList<>();
            this.weight = weight;
        }
        
        public void addSkill(Skill skill) {
            skills.add(skill);
        }
        
        // Getters
        public String getName() { return name; }
        public String getDescription() { return description; }
        public List<Skill> getSkills() { return skills; }
        public double getWeight() { return weight; }
    }
    
    /**
     * Represents an individual skill
     */
    public static class Skill {
        private final String name;
        private final String description;
        private final SkillLevel currentLevel;
        private final SkillLevel targetLevel;
        private final double importance; // 0.0 to 1.0
        
        public Skill(String name, String description, SkillLevel currentLevel, 
                    SkillLevel targetLevel, double importance) {
            this.name = name;
            this.description = description;
            this.currentLevel = currentLevel;
            this.targetLevel = targetLevel;
            this.importance = importance;
        }
        
        // Getters
        public String getName() { return name; }
        public String getDescription() { return description; }
        public SkillLevel getCurrentLevel() { return currentLevel; }
        public SkillLevel getTargetLevel() { return targetLevel; }
        public double getImportance() { return importance; }
        
        /**
         * Calculate skill gap
         */
        public int getSkillGap() {
            return targetLevel.ordinal() - currentLevel.ordinal();
        }
        
        /**
         * Calculate priority score (gap * importance)
         */
        public double getPriorityScore() {
            return getSkillGap() * importance;
        }
    }
    
    /**
     * Represents skill levels
     */
    public enum SkillLevel {
        BEGINNER(1, "Beginner", "Basic understanding and ability to use with guidance"),
        INTERMEDIATE(2, "Intermediate", "Can work independently with occasional guidance"),
        ADVANCED(3, "Advanced", "Deep understanding and ability to solve complex problems"),
        EXPERT(4, "Expert", "Mastery level with ability to innovate and teach others"),
        LEADER(5, "Leader", "Industry thought leader and innovator");
        
        private final int level;
        private final String name;
        private final String description;
        
        SkillLevel(int level, String name, String description) {
            this.level = level;
            this.name = name;
            this.description = description;
        }
        
        public int getLevel() { return level; }
        public String getName() { return name; }
        public String getDescription() { return description; }
    }
    
    /**
     * Represents an assessment result
     */
    public static class AssessmentResult {
        private final String assessmentId;
        private final Date timestamp;
        private final Map<String, SkillCategory> skillCategories;
        private final double overallScore;
        private final List<Skill> topPriorities;
        private final Map<String, Double> categoryScores;
        
        public AssessmentResult(String assessmentId, Map<String, SkillCategory> skillCategories,
                              double overallScore, List<Skill> topPriorities, 
                              Map<String, Double> categoryScores) {
            this.assessmentId = assessmentId;
            this.timestamp = new Date();
            this.skillCategories = new HashMap<>(skillCategories);
            this.overallScore = overallScore;
            this.topPriorities = new ArrayList<>(topPriorities);
            this.categoryScores = new HashMap<>(categoryScores);
        }
        
        // Getters
        public String getAssessmentId() { return assessmentId; }
        public Date getTimestamp() { return timestamp; }
        public Map<String, SkillCategory> getSkillCategories() { return skillCategories; }
        public double getOverallScore() { return overallScore; }
        public List<Skill> getTopPriorities() { return topPriorities; }
        public Map<String, Double> getCategoryScores() { return categoryScores; }
    }
    
    /**
     * Initialize skill categories with predefined skills
     */
    private void initializeSkillCategories() {
        // Programming and Technical Skills
        SkillCategory programming = new SkillCategory("Programming", 
            "Core programming skills essential for AI development", 0.25);
        
        programming.addSkill(new Skill("Java Programming", "Proficiency in Java language and ecosystem", 
            SkillLevel.BEGINNER, SkillLevel.EXPERT, 0.9));
        programming.addSkill(new Skill("Python Programming", "Proficiency in Python for AI/ML", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.8));
        programming.addSkill(new Skill("Data Structures & Algorithms", "Understanding of fundamental algorithms", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.7));
        programming.addSkill(new Skill("Software Engineering", "Best practices and design patterns", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.6));
        
        skillCategories.put("programming", programming);
        
        // Machine Learning Skills
        SkillCategory machineLearning = new SkillCategory("Machine Learning", 
            "Core machine learning concepts and techniques", 0.30);
        
        machineLearning.addSkill(new Skill("Supervised Learning", "Classification and regression techniques", 
            SkillLevel.BEGINNER, SkillLevel.EXPERT, 0.9));
        machineLearning.addSkill(new Skill("Unsupervised Learning", "Clustering and dimensionality reduction", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.8));
        machineLearning.addSkill(new Skill("Deep Learning", "Neural networks and deep architectures", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.8));
        machineLearning.addSkill(new Skill("Model Evaluation", "Cross-validation and performance metrics", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.7));
        machineLearning.addSkill(new Skill("Feature Engineering", "Data preprocessing and feature selection", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.7));
        
        skillCategories.put("machineLearning", machineLearning);
        
        // Data Science Skills
        SkillCategory dataScience = new SkillCategory("Data Science", 
            "Data analysis and statistical skills", 0.20);
        
        dataScience.addSkill(new Skill("Statistics", "Statistical concepts and hypothesis testing", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.8));
        dataScience.addSkill(new Skill("Data Visualization", "Creating effective visualizations", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.6));
        dataScience.addSkill(new Skill("Data Wrangling", "Cleaning and preparing data", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.7));
        dataScience.addSkill(new Skill("SQL", "Database querying and management", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.5));
        
        skillCategories.put("dataScience", dataScience);
        
        // AI Ethics and Responsible AI
        SkillCategory ethics = new SkillCategory("AI Ethics", 
            "Understanding of ethical considerations in AI", 0.15);
        
        ethics.addSkill(new Skill("Bias Detection", "Identifying and mitigating bias in AI systems", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.8));
        ethics.addSkill(new Skill("Fairness Metrics", "Measuring and ensuring fairness", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.7));
        ethics.addSkill(new Skill("Privacy Preservation", "Techniques for protecting privacy", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.6));
        ethics.addSkill(new Skill("Explainable AI", "Making AI decisions interpretable", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.6));
        
        skillCategories.put("ethics", ethics);
        
        // Domain Knowledge
        SkillCategory domainKnowledge = new SkillCategory("Domain Knowledge", 
            "Industry-specific knowledge and applications", 0.10);
        
        domainKnowledge.addSkill(new Skill("Healthcare AI", "AI applications in healthcare", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.5));
        domainKnowledge.addSkill(new Skill("Financial AI", "AI in finance and fintech", 
            SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 0.5));
        domainKnowledge.addSkill(new Skill("Computer Vision", "Image and video processing", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.6));
        domainKnowledge.addSkill(new Skill("NLP", "Natural language processing", 
            SkillLevel.BEGINNER, SkillLevel.ADVANCED, 0.6));
        
        skillCategories.put("domainKnowledge", domainKnowledge);
    }
    
    /**
     * Conduct a comprehensive skills assessment
     * 
     * @return Assessment result with detailed analysis
     */
    public AssessmentResult conductAssessment() {
        logger.info("Starting comprehensive skills assessment");
        
        String assessmentId = "ASSESSMENT_" + System.currentTimeMillis();
        Map<String, Double> categoryScores = new HashMap<>();
        List<Skill> allSkills = new ArrayList<>();
        
        // Calculate scores for each category
        for (Map.Entry<String, SkillCategory> entry : skillCategories.entrySet()) {
            SkillCategory category = entry.getValue();
            double categoryScore = calculateCategoryScore(category);
            categoryScores.put(category.getName(), categoryScore);
            allSkills.addAll(category.getSkills());
        }
        
        // Calculate overall score
        double overallScore = calculateOverallScore(categoryScores);
        
        // Identify top priorities
        List<Skill> topPriorities = identifyTopPriorities(allSkills);
        
        // Create assessment result
        AssessmentResult result = new AssessmentResult(assessmentId, skillCategories, 
                                                     overallScore, topPriorities, categoryScores);
        
        // Store in history
        assessmentHistory.add(result);
        
        logger.info("Assessment completed. Overall score: {:.2f}", overallScore);
        return result;
    }
    
    /**
     * Calculate score for a skill category
     */
    private double calculateCategoryScore(SkillCategory category) {
        if (category.getSkills().isEmpty()) {
            return 0.0;
        }
        
        double totalScore = 0.0;
        double totalWeight = 0.0;
        
        for (Skill skill : category.getSkills()) {
            double skillScore = (double) skill.getCurrentLevel().getLevel() / SkillLevel.LEADER.getLevel();
            double weightedScore = skillScore * skill.getImportance();
            totalScore += weightedScore;
            totalWeight += skill.getImportance();
        }
        
        return totalWeight > 0 ? totalScore / totalWeight : 0.0;
    }
    
    /**
     * Calculate overall assessment score
     */
    private double calculateOverallScore(Map<String, Double> categoryScores) {
        double totalScore = 0.0;
        double totalWeight = 0.0;
        
        for (Map.Entry<String, Double> entry : categoryScores.entrySet()) {
            String categoryName = entry.getKey();
            double score = entry.getValue();
            
            SkillCategory category = skillCategories.values().stream()
                .filter(c -> c.getName().equals(categoryName))
                .findFirst()
                .orElse(null);
            
            if (category != null) {
                totalScore += score * category.getWeight();
                totalWeight += category.getWeight();
            }
        }
        
        return totalWeight > 0 ? totalScore / totalWeight : 0.0;
    }
    
    /**
     * Identify top priority skills for improvement
     */
    private List<Skill> identifyTopPriorities(List<Skill> allSkills) {
        return allSkills.stream()
            .filter(skill -> skill.getSkillGap() > 0) // Only skills that need improvement
            .sorted(Comparator.comparing(Skill::getPriorityScore).reversed())
            .limit(10) // Top 10 priorities
            .collect(Collectors.toList());
    }
    
    /**
     * Update skill level for a specific skill
     */
    public void updateSkillLevel(String categoryName, String skillName, SkillLevel newLevel) {
        SkillCategory category = skillCategories.get(categoryName);
        if (category != null) {
            for (Skill skill : category.getSkills()) {
                if (skill.getName().equals(skillName)) {
                    // Create new skill with updated level
                    Skill updatedSkill = new Skill(skill.getName(), skill.getDescription(),
                                                  newLevel, skill.getTargetLevel(), skill.getImportance());
                    
                    // Replace the skill in the category
                    category.getSkills().remove(skill);
                    category.addSkill(updatedSkill);
                    
                    logger.info("Updated skill level: {} - {} to {}", categoryName, skillName, newLevel.getName());
                    return;
                }
            }
        }
        
        logger.warn("Skill not found: {} in category {}", skillName, categoryName);
    }
    
    /**
     * Generate learning recommendations based on assessment
     */
    public List<String> generateLearningRecommendations(AssessmentResult result) {
        List<String> recommendations = new ArrayList<>();
        
        // Overall recommendations
        if (result.getOverallScore() < 0.3) {
            recommendations.add("Focus on building foundational skills before advancing to complex topics");
        } else if (result.getOverallScore() < 0.6) {
            recommendations.add("Strengthen intermediate skills and start working on advanced concepts");
        } else {
            recommendations.add("Focus on specialization and leadership skills");
        }
        
        // Category-specific recommendations
        for (Map.Entry<String, Double> entry : result.getCategoryScores().entrySet()) {
            String categoryName = entry.getKey();
            double score = entry.getValue();
            
            if (score < 0.4) {
                recommendations.add("Prioritize learning in " + categoryName + " category");
            }
        }
        
        // Skill-specific recommendations
        for (Skill skill : result.getTopPriorities()) {
            if (skill.getSkillGap() >= 2) {
                recommendations.add("Consider formal training or certification for " + skill.getName());
            } else if (skill.getSkillGap() == 1) {
                recommendations.add("Practice and apply " + skill.getName() + " in real projects");
            }
        }
        
        return recommendations;
    }
    
    /**
     * Generate detailed assessment report
     */
    public String generateAssessmentReport(AssessmentResult result) {
        StringBuilder report = new StringBuilder();
        report.append("=== AI Skills Assessment Report ===\n");
        report.append("Assessment ID: ").append(result.getAssessmentId()).append("\n");
        report.append("Date: ").append(result.getTimestamp()).append("\n");
        report.append("Overall Score: ").append(String.format("%.2f", result.getOverallScore())).append("\n\n");
        
        // Category scores
        report.append("Category Scores:\n");
        for (Map.Entry<String, Double> entry : result.getCategoryScores().entrySet()) {
            report.append("  ").append(entry.getKey()).append(": ")
                  .append(String.format("%.2f", entry.getValue())).append("\n");
        }
        report.append("\n");
        
        // Top priorities
        report.append("Top Learning Priorities:\n");
        for (int i = 0; i < result.getTopPriorities().size(); i++) {
            Skill skill = result.getTopPriorities().get(i);
            report.append("  ").append(i + 1).append(". ").append(skill.getName())
                  .append(" (Gap: ").append(skill.getSkillGap())
                  .append(", Priority: ").append(String.format("%.2f", skill.getPriorityScore()))
                  .append(")\n");
        }
        report.append("\n");
        
        // Recommendations
        report.append("Learning Recommendations:\n");
        List<String> recommendations = generateLearningRecommendations(result);
        for (String recommendation : recommendations) {
            report.append("  - ").append(recommendation).append("\n");
        }
        
        return report.toString();
    }
    
    /**
     * Get assessment history
     */
    public List<AssessmentResult> getAssessmentHistory() {
        return new ArrayList<>(assessmentHistory);
    }
    
    /**
     * Get skill categories
     */
    public Map<String, SkillCategory> getSkillCategories() {
        return new HashMap<>(skillCategories);
    }
}

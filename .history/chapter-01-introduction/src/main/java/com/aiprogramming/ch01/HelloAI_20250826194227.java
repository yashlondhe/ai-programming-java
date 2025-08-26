package com.aiprogramming.ch01;

/**
 * Hello AI World - Your first AI program in Java
 * 
 * This example introduces basic AI concepts and demonstrates
 * how to structure AI applications in Java.
 */
public class HelloAI {
    
    public static void main(String[] args) {
        System.out.println("🤖 Welcome to AI Programming with Java!");
        
        // Demonstrate basic AI concepts
        demonstrateAIConcepts();
        
        // Show Java AI ecosystem
        exploreJavaAIEcosystem();
        
        // Run a simple AI simulation
        runSimpleAISimulation();
        
        System.out.println("🎉 Congratulations! You've run your first AI program!");
    }
    
    /**
     * Demonstrates fundamental AI concepts with simple examples
     */
    private static void demonstrateAIConcepts() {
        System.out.println("\n📚 AI Concepts Demonstration:");
        
        // 1. Pattern Recognition
        System.out.println("1. Pattern Recognition: AI can identify patterns in data");
        String[] patterns = {"spam", "not spam", "spam", "not spam", "spam"};
        String prediction = predictNextPattern(patterns);
        System.out.println("   Prediction for next email: " + prediction);
        
        // 2. Decision Making
        System.out.println("2. Decision Making: AI can make decisions based on rules");
        double temperature = 25.0;
        String decision = makeDecision(temperature);
        System.out.println("   Decision for temperature " + temperature + "°C: " + decision);
        
        // 3. Learning from Data
        System.out.println("3. Learning from Data: AI improves with more information");
        int[] learningProgress = {60, 65, 70, 75, 80, 85, 90};
        double improvement = calculateImprovement(learningProgress);
        System.out.printf("   Learning improvement: %.2f%%\n", improvement);
    }
    
    /**
     * Explores the Java AI ecosystem and available libraries
     */
    private static void exploreJavaAIEcosystem() {
        System.out.println("\n🛠️ Java AI Ecosystem:");
        
        String[] libraries = {
            "DeepLearning4J (DL4J) - Deep learning for Java",
            "Tribuo - Machine learning library by Oracle",
            "Smile - Statistical machine learning",
            "OpenNLP - Natural language processing",
            "Stanford NLP - Advanced NLP tools",
            "ND4J - Numerical computing for Java"
        };
        
        for (int i = 0; i < libraries.length; i++) {
            System.out.println((i + 1) + ". " + libraries[i]);
        }
        
        System.out.println("\n💡 These libraries will be covered in detail throughout the book!");
    }
    
    /**
     * Runs a simple AI simulation to demonstrate basic concepts
     */
    private static void runSimpleAISimulation() {
        System.out.println("\n🎮 Simple AI Simulation:");
        
        // Simulate a simple chatbot
        String[] userInputs = {"Hello", "How are you?", "What's the weather?", "Goodbye"};
        
        for (String input : userInputs) {
            String response = generateResponse(input);
            System.out.println("User: " + input);
            System.out.println("AI:  " + response);
        }
    }
    
    // Helper methods for demonstrations
    
    private static String predictNextPattern(String[] patterns) {
        // Simple pattern recognition: count occurrences
        int spamCount = 0;
        for (String pattern : patterns) {
            if ("spam".equals(pattern)) {
                spamCount++;
            }
        }
        
        // If more than 50% are spam, predict spam
        return spamCount > patterns.length / 2 ? "spam" : "not spam";
    }
    
    private static String makeDecision(double temperature) {
        // Simple rule-based decision making
        if (temperature < 0) {
            return "Turn on heating";
        } else if (temperature > 30) {
            return "Turn on air conditioning";
        } else {
            return "Maintain current temperature";
        }
    }
    
    private static double calculateImprovement(int[] progress) {
        if (progress.length < 2) {
            return 0.0;
        }
        
        int initial = progress[0];
        int final_score = progress[progress.length - 1];
        
        return ((double) (final_score - initial) / initial) * 100;
    }
    
    private static String generateResponse(String input) {
        // Simple rule-based response generation
        String lowerInput = input.toLowerCase();
        
        if (lowerInput.contains("hello") || lowerInput.contains("hi")) {
            return "Hello! How can I help you today?";
        } else if (lowerInput.contains("how are you")) {
            return "I'm functioning perfectly! Thank you for asking.";
        } else if (lowerInput.contains("weather")) {
            return "I'm sorry, I don't have access to real-time weather data yet.";
        } else if (lowerInput.contains("goodbye") || lowerInput.contains("bye")) {
            return "Goodbye! Have a great day!";
        } else {
            return "I'm still learning. Could you rephrase that?";
        }
    }
}

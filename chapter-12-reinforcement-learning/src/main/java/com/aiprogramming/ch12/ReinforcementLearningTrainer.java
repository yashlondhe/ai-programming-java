package com.aiprogramming.ch12;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for training reinforcement learning agents
 */
public class ReinforcementLearningTrainer {
    
    /**
     * Train an agent in an environment for a specified number of episodes
     */
    public static TrainingResult trainAgent(Agent agent, Environment environment, 
                                          int numEpisodes, boolean verbose) {
        List<Double> episodeRewards = new ArrayList<>();
        List<Integer> episodeLengths = new ArrayList<>();
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            double totalReward = 0.0;
            int stepCount = 0;
            
            // Reset environment and agent
            int currentState = environment.reset();
            agent.reset();
            
            while (!environment.isDone() && stepCount < 1000) { // Max 1000 steps per episode
                // Agent selects action
                int action = agent.selectAction(currentState);
                
                // Take action in environment
                StepResult result = environment.step(action);
                
                // Agent learns from experience
                agent.learn(currentState, action, result.getReward(), 
                           result.getNextState(), result.isDone());
                
                // Update for next step
                currentState = result.getNextState();
                totalReward += result.getReward();
                stepCount++;
            }
            
            // Update agent after episode
            agent.updateAfterEpisode();
            
            // Record episode statistics
            episodeRewards.add(totalReward);
            episodeLengths.add(stepCount);
            
            // Print progress
            if (verbose && episode % 100 == 0) {
                System.out.printf("Episode %d: Reward = %.2f, Steps = %d, Exploration = %.3f%n", 
                                episode, totalReward, stepCount, agent.getExplorationRate());
            }
        }
        
        return new TrainingResult(episodeRewards, episodeLengths);
    }
    
    /**
     * Evaluate a trained agent's performance
     */
    public static EvaluationResult evaluateAgent(Agent agent, Environment environment, 
                                               int numEpisodes) {
        List<Double> episodeRewards = new ArrayList<>();
        List<Integer> episodeLengths = new ArrayList<>();
        
        // Save current exploration rate and set to 0 for evaluation
        double originalExplorationRate = agent.getExplorationRate();
        agent.setExplorationRate(0.0);
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            double totalReward = 0.0;
            int stepCount = 0;
            
            int currentState = environment.reset();
            
            while (!environment.isDone() && stepCount < 1000) {
                int action = agent.selectAction(currentState);
                StepResult result = environment.step(action);
                
                currentState = result.getNextState();
                totalReward += result.getReward();
                stepCount++;
            }
            
            episodeRewards.add(totalReward);
            episodeLengths.add(stepCount);
        }
        
        // Restore original exploration rate
        agent.setExplorationRate(originalExplorationRate);
        
        return new EvaluationResult(episodeRewards, episodeLengths);
    }
    
    /**
     * Run a single episode and visualize the agent's behavior
     */
    public static void demonstrateAgent(Agent agent, Environment environment, 
                                      boolean render, boolean useGreedyPolicy) {
        // Save current exploration rate
        double originalExplorationRate = agent.getExplorationRate();
        
        if (useGreedyPolicy) {
            agent.setExplorationRate(0.0); // Use greedy policy
        }
        
        int currentState = environment.reset();
        double totalReward = 0.0;
        int stepCount = 0;
        
        System.out.println("=== Agent Demonstration ===");
        
        if (render) {
            environment.render();
        }
        
        while (!environment.isDone() && stepCount < 100) {
            int action = agent.selectAction(currentState);
            
            System.out.printf("Step %d: State = %d, Action = %s%n", 
                            stepCount, currentState, actionToString(action));
            
            StepResult result = environment.step(action);
            
            currentState = result.getNextState();
            totalReward += result.getReward();
            stepCount++;
            
            if (render) {
                System.out.printf("Reward: %.2f, Next State: %d%n", 
                                result.getReward(), result.getNextState());
                environment.render();
                
                try {
                    Thread.sleep(1000); // Pause for visualization
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        
        System.out.printf("Episode finished: Total Reward = %.2f, Steps = %d%n", 
                        totalReward, stepCount);
        
        // Restore original exploration rate
        agent.setExplorationRate(originalExplorationRate);
    }
    
    /**
     * Compare multiple agents on the same environment
     */
    public static void compareAgents(List<Agent> agents, List<String> agentNames, 
                                   Environment environment, int numEpisodes) {
        System.out.println("=== Agent Comparison ===");
        
        for (int i = 0; i < agents.size(); i++) {
            Agent agent = agents.get(i);
            String name = agentNames.get(i);
            
            System.out.printf("\nTraining %s...\n", name);
            TrainingResult result = trainAgent(agent, environment, numEpisodes, false);
            
            double avgReward = result.getAverageReward();
            double avgLength = result.getAverageLength();
            
            System.out.printf("%s Results: Avg Reward = %.3f, Avg Length = %.1f%n", 
                            name, avgReward, avgLength);
        }
    }
    
    /**
     * Convert action to string for display
     */
    private static String actionToString(int action) {
        switch (action) {
            case GridWorld.ACTION_UP: return "UP";
            case GridWorld.ACTION_DOWN: return "DOWN";
            case GridWorld.ACTION_LEFT: return "LEFT";
            case GridWorld.ACTION_RIGHT: return "RIGHT";
            default: return "ACTION_" + action;
        }
    }
    
    /**
     * Container class for training results
     */
    public static class TrainingResult {
        private final List<Double> episodeRewards;
        private final List<Integer> episodeLengths;
        
        public TrainingResult(List<Double> episodeRewards, List<Integer> episodeLengths) {
            this.episodeRewards = new ArrayList<>(episodeRewards);
            this.episodeLengths = new ArrayList<>(episodeLengths);
        }
        
        public List<Double> getEpisodeRewards() {
            return new ArrayList<>(episodeRewards);
        }
        
        public List<Integer> getEpisodeLengths() {
            return new ArrayList<>(episodeLengths);
        }
        
        public double getAverageReward() {
            return episodeRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        }
        
        public double getAverageLength() {
            return episodeLengths.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        }
        
        public double getTotalReward() {
            return episodeRewards.stream().mapToDouble(Double::doubleValue).sum();
        }
        
        public void printSummary() {
            System.out.printf("Training Summary: %d episodes%n", episodeRewards.size());
            System.out.printf("Average Reward: %.3f%n", getAverageReward());
            System.out.printf("Average Length: %.1f%n", getAverageLength());
            System.out.printf("Total Reward: %.2f%n", getTotalReward());
        }
    }
    
    /**
     * Container class for evaluation results
     */
    public static class EvaluationResult {
        private final List<Double> episodeRewards;
        private final List<Integer> episodeLengths;
        
        public EvaluationResult(List<Double> episodeRewards, List<Integer> episodeLengths) {
            this.episodeRewards = new ArrayList<>(episodeRewards);
            this.episodeLengths = new ArrayList<>(episodeLengths);
        }
        
        public List<Double> getEpisodeRewards() {
            return new ArrayList<>(episodeRewards);
        }
        
        public List<Integer> getEpisodeLengths() {
            return new ArrayList<>(episodeLengths);
        }
        
        public double getAverageReward() {
            return episodeRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        }
        
        public double getAverageLength() {
            return episodeLengths.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        }
        
        public double getSuccessRate() {
            // For GridWorld, success is positive reward
            long successes = episodeRewards.stream().mapToLong(r -> r > 0 ? 1 : 0).sum();
            return (double) successes / episodeRewards.size();
        }
        
        public void printSummary() {
            System.out.printf("Evaluation Summary: %d episodes%n", episodeRewards.size());
            System.out.printf("Average Reward: %.3f%n", getAverageReward());
            System.out.printf("Average Length: %.1f%n", getAverageLength());
            System.out.printf("Success Rate: %.1f%%%n", getSuccessRate() * 100);
        }
    }
}

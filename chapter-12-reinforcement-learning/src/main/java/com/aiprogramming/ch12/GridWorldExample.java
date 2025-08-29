package com.aiprogramming.ch12;

import java.util.Arrays;

/**
 * Comprehensive example demonstrating reinforcement learning algorithms
 * on a GridWorld environment
 */
public class GridWorldExample {
    
    public static void main(String[] args) {
        System.out.println("=== Reinforcement Learning in Grid World ===\n");
        
        // Create a simple grid world
        GridWorld environment = GridWorld.createSimpleGridWorld();
        
        System.out.println("Grid World Environment:");
        environment.reset();
        environment.render();
        
        // Demonstrate different algorithms
        demonstrateQLearning(environment);
        demonstrateSarsa(environment);
        demonstrateValueIteration(environment);
        demonstratePolicyIteration(environment);
        
        // Compare algorithms
        compareAlgorithms(environment);
    }
    
    /**
     * Demonstrate Q-Learning algorithm
     */
    private static void demonstrateQLearning(GridWorld environment) {
        System.out.println("=== Q-Learning Demonstration ===");
        
        // Create Q-Learning agent
        QLearningAgent qAgent = new QLearningAgent(
            environment.getStateSize(),
            environment.getActionSize(),
            0.1,    // learning rate
            0.95,   // discount factor
            0.9     // exploration rate
        );
        
        // Train the agent
        System.out.println("Training Q-Learning agent...");
        ReinforcementLearningTrainer.TrainingResult qResult = 
            ReinforcementLearningTrainer.trainAgent(qAgent, environment, 1000, true);
        
        System.out.println("\nQ-Learning Training Results:");
        qResult.printSummary();
        
        // Evaluate the trained agent
        System.out.println("\nEvaluating Q-Learning agent:");
        ReinforcementLearningTrainer.EvaluationResult qEvaluation = 
            ReinforcementLearningTrainer.evaluateAgent(qAgent, environment, 100);
        qEvaluation.printSummary();
        
        // Show learned Q-table
        System.out.println("\nLearned Q-Table:");
        qAgent.printQTable();
        
        // Demonstrate learned policy
        System.out.println("Demonstrating learned policy:");
        ReinforcementLearningTrainer.demonstrateAgent(qAgent, environment, true, true);
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * Demonstrate SARSA algorithm
     */
    private static void demonstrateSarsa(GridWorld environment) {
        System.out.println("=== SARSA Demonstration ===");
        
        // Create SARSA agent
        SarsaAgent sarsaAgent = new SarsaAgent(
            environment.getStateSize(),
            environment.getActionSize(),
            0.1,    // learning rate
            0.95,   // discount factor
            0.9     // exploration rate
        );
        
        // Train the agent
        System.out.println("Training SARSA agent...");
        ReinforcementLearningTrainer.TrainingResult sarsaResult = 
            ReinforcementLearningTrainer.trainAgent(sarsaAgent, environment, 1000, true);
        
        System.out.println("\nSARSA Training Results:");
        sarsaResult.printSummary();
        
        // Evaluate the trained agent
        System.out.println("\nEvaluating SARSA agent:");
        ReinforcementLearningTrainer.EvaluationResult sarsaEvaluation = 
            ReinforcementLearningTrainer.evaluateAgent(sarsaAgent, environment, 100);
        sarsaEvaluation.printSummary();
        
        // Show learned Q-table
        System.out.println("\nLearned Q-Table:");
        sarsaAgent.printQTable();
        
        // Demonstrate learned policy
        System.out.println("Demonstrating learned policy:");
        ReinforcementLearningTrainer.demonstrateAgent(sarsaAgent, environment, true, true);
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * Demonstrate Value Iteration algorithm
     */
    private static void demonstrateValueIteration(GridWorld environment) {
        System.out.println("=== Value Iteration Demonstration ===");
        
        // Create Value Iteration solver
        ValueIteration valueIteration = new ValueIteration(
            environment,
            0.95,   // discount factor
            0.001,  // tolerance
            1000    // max iterations
        );
        
        // Solve the MDP
        System.out.println("Running Value Iteration...");
        valueIteration.solve();
        
        // Print results
        valueIteration.printResults();
        
        // Create an agent that follows the optimal policy
        OptimalPolicyAgent optimalAgent = new OptimalPolicyAgent(
            valueIteration.getPolicy(), valueIteration.getValues());
        
        // Demonstrate optimal policy
        System.out.println("Demonstrating optimal policy from Value Iteration:");
        ReinforcementLearningTrainer.demonstrateAgent(optimalAgent, environment, true, true);
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * Demonstrate Policy Iteration algorithm
     */
    private static void demonstratePolicyIteration(GridWorld environment) {
        System.out.println("=== Policy Iteration Demonstration ===");
        
        // Create Policy Iteration solver
        PolicyIteration policyIteration = new PolicyIteration(
            environment,
            0.95,   // discount factor
            0.001,  // tolerance
            100     // max iterations
        );
        
        // Solve the MDP
        System.out.println("Running Policy Iteration...");
        policyIteration.solve();
        
        // Print results
        policyIteration.printResults();
        
        // Create an agent that follows the optimal policy
        OptimalPolicyAgent optimalAgent = new OptimalPolicyAgent(
            policyIteration.getPolicy(), policyIteration.getValues());
        
        // Demonstrate optimal policy
        System.out.println("Demonstrating optimal policy from Policy Iteration:");
        ReinforcementLearningTrainer.demonstrateAgent(optimalAgent, environment, true, true);
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * Compare different algorithms
     */
    private static void compareAlgorithms(GridWorld environment) {
        System.out.println("=== Algorithm Comparison ===");
        
        // Create agents
        QLearningAgent qAgent = new QLearningAgent(
            environment.getStateSize(), environment.getActionSize(), 0.1, 0.95, 0.9);
        
        SarsaAgent sarsaAgent = new SarsaAgent(
            environment.getStateSize(), environment.getActionSize(), 0.1, 0.95, 0.9);
        
        // Compare performance
        ReinforcementLearningTrainer.compareAgents(
            Arrays.asList(qAgent, sarsaAgent),
            Arrays.asList("Q-Learning", "SARSA"),
            environment,
            500
        );
        
        System.out.println("\nComparison complete!");
    }
    
    /**
     * Simple agent that follows a fixed optimal policy
     */
    private static class OptimalPolicyAgent extends Agent {
        private final int[] policy;
        private final double[] values;
        
        public OptimalPolicyAgent(int[] policy, double[] values) {
            super(policy.length, 4, 0.0, 0.0);
            this.policy = policy.clone();
            this.values = values.clone();
        }
        
        @Override
        public int selectAction(int state) {
            return policy[state] >= 0 ? policy[state] : 0;
        }
        
        @Override
        public void learn(int state, int action, double reward, int nextState, boolean done) {
            // Optimal policy doesn't need to learn
        }
        
        @Override
        public double getQValue(int state, int action) {
            return action == policy[state] ? values[state] : 0.0;
        }
        
        @Override
        public double getStateValue(int state) {
            return values[state];
        }
        
        @Override
        public void setExplorationRate(double explorationRate) {
            // Optimal policy doesn't explore
        }
        
        @Override
        public double getExplorationRate() {
            return 0.0;
        }
    }
}

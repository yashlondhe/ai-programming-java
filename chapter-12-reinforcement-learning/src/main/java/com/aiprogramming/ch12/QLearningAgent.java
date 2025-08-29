package com.aiprogramming.ch12;

import java.util.Random;

/**
 * Q-Learning agent implementation
 * Q-Learning is a model-free reinforcement learning algorithm
 */
public class QLearningAgent extends Agent {
    
    private final double[][] qTable;
    private double explorationRate;
    private double explorationDecay;
    private double minExplorationRate;
    private final Random random;
    
    public QLearningAgent(int stateSize, int actionSize, double learningRate, 
                         double discountFactor, double explorationRate) {
        super(stateSize, actionSize, learningRate, discountFactor);
        this.qTable = new double[stateSize][actionSize];
        this.explorationRate = explorationRate;
        this.explorationDecay = 0.995;
        this.minExplorationRate = 0.01;
        this.random = new Random();
        
        // Initialize Q-table with zeros
        initializeQTable();
    }
    
    public QLearningAgent(int stateSize, int actionSize, double learningRate, 
                         double discountFactor, double explorationRate, 
                         double explorationDecay, double minExplorationRate) {
        this(stateSize, actionSize, learningRate, discountFactor, explorationRate);
        this.explorationDecay = explorationDecay;
        this.minExplorationRate = minExplorationRate;
    }
    
    private void initializeQTable() {
        for (int state = 0; state < stateSize; state++) {
            for (int action = 0; action < actionSize; action++) {
                qTable[state][action] = 0.0;
            }
        }
    }
    
    @Override
    public int selectAction(int state) {
        // Epsilon-greedy action selection
        if (random.nextDouble() < explorationRate) {
            // Explore: choose random action
            return random.nextInt(actionSize);
        } else {
            // Exploit: choose action with highest Q-value
            return getGreedyAction(state);
        }
    }
    
    /**
     * Get the greedy action (action with highest Q-value) for a given state
     */
    public int getGreedyAction(int state) {
        int bestAction = 0;
        double bestValue = qTable[state][0];
        
        for (int action = 1; action < actionSize; action++) {
            if (qTable[state][action] > bestValue) {
                bestValue = qTable[state][action];
                bestAction = action;
            }
        }
        
        return bestAction;
    }
    
    @Override
    public void learn(int state, int action, double reward, int nextState, boolean done) {
        // Q-Learning update rule:
        // Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        
        double currentQValue = qTable[state][action];
        double maxNextQValue = done ? 0.0 : getMaxQValue(nextState);
        
        double targetValue = reward + discountFactor * maxNextQValue;
        double tdError = targetValue - currentQValue;
        
        qTable[state][action] = currentQValue + learningRate * tdError;
    }
    
    /**
     * Get the maximum Q-value for a given state
     */
    public double getMaxQValue(int state) {
        double maxValue = qTable[state][0];
        for (int action = 1; action < actionSize; action++) {
            maxValue = Math.max(maxValue, qTable[state][action]);
        }
        return maxValue;
    }
    
    @Override
    public void updateAfterEpisode() {
        // Decay exploration rate
        explorationRate = Math.max(minExplorationRate, 
                                 explorationRate * explorationDecay);
    }
    
    @Override
    public double getQValue(int state, int action) {
        return qTable[state][action];
    }
    
    @Override
    public double getStateValue(int state) {
        return getMaxQValue(state);
    }
    
    @Override
    public void setExplorationRate(double explorationRate) {
        this.explorationRate = explorationRate;
    }
    
    @Override
    public double getExplorationRate() {
        return explorationRate;
    }
    
    /**
     * Print the Q-table (useful for debugging and analysis)
     */
    public void printQTable() {
        System.out.println("Q-Table:");
        System.out.printf("%8s", "State");
        for (int action = 0; action < actionSize; action++) {
            System.out.printf("%12s%d", "Action", action);
        }
        System.out.println();
        
        for (int state = 0; state < stateSize; state++) {
            System.out.printf("%8d", state);
            for (int action = 0; action < actionSize; action++) {
                System.out.printf("%12.3f", qTable[state][action]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    /**
     * Get a copy of the Q-table
     */
    public double[][] getQTable() {
        double[][] copy = new double[stateSize][actionSize];
        for (int state = 0; state < stateSize; state++) {
            System.arraycopy(qTable[state], 0, copy[state], 0, actionSize);
        }
        return copy;
    }
    
    /**
     * Set exploration decay parameters
     */
    public void setExplorationDecay(double explorationDecay, double minExplorationRate) {
        this.explorationDecay = explorationDecay;
        this.minExplorationRate = minExplorationRate;
    }
    
    /**
     * Initialize Q-table with random values (sometimes helpful for exploration)
     */
    public void initializeRandomQTable(double min, double max) {
        for (int state = 0; state < stateSize; state++) {
            for (int action = 0; action < actionSize; action++) {
                qTable[state][action] = min + random.nextDouble() * (max - min);
            }
        }
    }
}

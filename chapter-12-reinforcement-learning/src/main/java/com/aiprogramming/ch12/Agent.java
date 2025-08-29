package com.aiprogramming.ch12;

/**
 * Abstract base class for reinforcement learning agents
 */
public abstract class Agent {
    protected int stateSize;
    protected int actionSize;
    protected double learningRate;
    protected double discountFactor;
    
    public Agent(int stateSize, int actionSize, double learningRate, double discountFactor) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
    }
    
    /**
     * Select an action given the current state
     * @param state current state
     * @return selected action
     */
    public abstract int selectAction(int state);
    
    /**
     * Learn from experience (state, action, reward, next_state)
     * @param state current state
     * @param action action taken
     * @param reward reward received
     * @param nextState next state reached
     * @param done whether the episode is finished
     */
    public abstract void learn(int state, int action, double reward, int nextState, boolean done);
    
    /**
     * Update the agent's parameters after an episode
     */
    public void updateAfterEpisode() {
        // Default implementation does nothing
        // Override in subclasses if needed
    }
    
    /**
     * Reset the agent's state for a new episode
     */
    public void reset() {
        // Default implementation does nothing
        // Override in subclasses if needed
    }
    
    /**
     * Get the action-value (Q-value) for a state-action pair
     * @param state the state
     * @param action the action
     * @return the Q-value
     */
    public abstract double getQValue(int state, int action);
    
    /**
     * Get the value of a state
     * @param state the state
     * @return the state value
     */
    public abstract double getStateValue(int state);
    
    /**
     * Set exploration parameters
     * @param explorationRate exploration rate (epsilon)
     */
    public abstract void setExplorationRate(double explorationRate);
    
    /**
     * Get current exploration rate
     * @return current exploration rate
     */
    public abstract double getExplorationRate();
}

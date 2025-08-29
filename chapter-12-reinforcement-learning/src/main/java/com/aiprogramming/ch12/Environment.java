package com.aiprogramming.ch12;

/**
 * Abstract base class for reinforcement learning environments
 * Follows the OpenAI Gym interface design
 */
public abstract class Environment {
    
    /**
     * Reset the environment to initial state
     * @return initial observation
     */
    public abstract int reset();
    
    /**
     * Take an action in the environment
     * @param action the action to take
     * @return step result containing next state, reward, done flag, and info
     */
    public abstract StepResult step(int action);
    
    /**
     * Get the current state of the environment
     * @return current state
     */
    public abstract int getCurrentState();
    
    /**
     * Get the number of possible states
     * @return number of states
     */
    public abstract int getStateSize();
    
    /**
     * Get the number of possible actions
     * @return number of actions
     */
    public abstract int getActionSize();
    
    /**
     * Check if the current episode is finished
     * @return true if episode is done
     */
    public abstract boolean isDone();
    
    /**
     * Render the environment (for visualization)
     */
    public abstract void render();
    
    /**
     * Check if the given state is terminal
     * @param state the state to check
     * @return true if the state is terminal
     */
    public abstract boolean isTerminalState(int state);
    
    /**
     * Get all possible actions from a given state
     * @param state the state
     * @return array of possible actions
     */
    public abstract int[] getPossibleActions(int state);
}

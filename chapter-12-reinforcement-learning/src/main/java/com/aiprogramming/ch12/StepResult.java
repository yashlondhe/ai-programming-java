package com.aiprogramming.ch12;

import java.util.Map;

/**
 * Container class for the result of taking a step in an environment
 */
public class StepResult {
    private final int nextState;
    private final double reward;
    private final boolean done;
    private final Map<String, Object> info;
    
    public StepResult(int nextState, double reward, boolean done, Map<String, Object> info) {
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
        this.info = info;
    }
    
    public StepResult(int nextState, double reward, boolean done) {
        this(nextState, reward, done, null);
    }
    
    public int getNextState() {
        return nextState;
    }
    
    public double getReward() {
        return reward;
    }
    
    public boolean isDone() {
        return done;
    }
    
    public Map<String, Object> getInfo() {
        return info;
    }
    
    @Override
    public String toString() {
        return String.format("StepResult{nextState=%d, reward=%.2f, done=%s}", 
                           nextState, reward, done);
    }
}

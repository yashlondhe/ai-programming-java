package com.aiprogramming.ch12;

/**
 * Value Iteration algorithm for solving Markov Decision Processes (MDPs)
 * This is a dynamic programming approach that computes optimal values and policy
 */
public class ValueIteration {
    
    private final Environment environment;
    private final double discountFactor;
    private final double tolerance;
    private final int maxIterations;
    
    private double[] values;
    private int[] policy;
    private boolean converged;
    private int iterationsUsed;
    
    public ValueIteration(Environment environment, double discountFactor, 
                         double tolerance, int maxIterations) {
        this.environment = environment;
        this.discountFactor = discountFactor;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        
        int stateSize = environment.getStateSize();
        this.values = new double[stateSize];
        this.policy = new int[stateSize];
        this.converged = false;
        this.iterationsUsed = 0;
    }
    
    /**
     * Run value iteration algorithm
     */
    public void solve() {
        int stateSize = environment.getStateSize();
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[] newValues = new double[stateSize];
            double maxChange = 0.0;
            
            // Update value for each state
            for (int state = 0; state < stateSize; state++) {
                if (environment.isTerminalState(state)) {
                    newValues[state] = getRewardForState(state);
                } else {
                    newValues[state] = computeOptimalValue(state);
                }
                
                double change = Math.abs(newValues[state] - values[state]);
                maxChange = Math.max(maxChange, change);
            }
            
            values = newValues;
            iterationsUsed = iteration + 1;
            
            // Check for convergence
            if (maxChange < tolerance) {
                converged = true;
                break;
            }
        }
        
        // Extract policy from values
        extractPolicy();
    }
    
    /**
     * Compute the optimal value for a state using the Bellman equation
     */
    private double computeOptimalValue(int state) {
        double maxValue = Double.NEGATIVE_INFINITY;
        int[] possibleActions = environment.getPossibleActions(state);
        
        for (int action : possibleActions) {
            double actionValue = computeActionValue(state, action);
            maxValue = Math.max(maxValue, actionValue);
        }
        
        return maxValue;
    }
    
    /**
     * Compute the value of taking a specific action in a state
     */
    private double computeActionValue(int state, int action) {
        // In a deterministic environment, this is straightforward
        // For stochastic environments, you'd sum over all possible outcomes
        
        // Simulate taking the action to get next state and reward
        int currentState = environment.getCurrentState();
        environment.reset();
        
        // Move to the target state (this is a simplification)
        // In practice, you'd have transition probabilities
        double reward = getRewardForStateAction(state, action);
        int nextState = getNextState(state, action);
        
        double actionValue = reward + discountFactor * values[nextState];
        
        return actionValue;
    }
    
    /**
     * Extract optimal policy from computed values
     */
    private void extractPolicy() {
        int stateSize = environment.getStateSize();
        
        for (int state = 0; state < stateSize; state++) {
            if (environment.isTerminalState(state)) {
                policy[state] = -1; // No action needed in terminal state
            } else {
                int bestAction = 0;
                double bestValue = Double.NEGATIVE_INFINITY;
                int[] possibleActions = environment.getPossibleActions(state);
                
                for (int action : possibleActions) {
                    double actionValue = computeActionValue(state, action);
                    if (actionValue > bestValue) {
                        bestValue = actionValue;
                        bestAction = action;
                    }
                }
                
                policy[state] = bestAction;
            }
        }
    }
    
    /**
     * Get the reward for being in a specific state
     */
    private double getRewardForState(int state) {
        if (environment instanceof GridWorld) {
            return ((GridWorld) environment).getReward(state);
        }
        return 0.0; // Default reward
    }
    
    /**
     * Get the reward for taking an action in a state
     */
    private double getRewardForStateAction(int state, int action) {
        // For GridWorld, reward depends on the resulting state
        int nextState = getNextState(state, action);
        return getRewardForState(nextState);
    }
    
    /**
     * Get the next state after taking an action (deterministic transition)
     */
    private int getNextState(int state, int action) {
        if (environment instanceof GridWorld) {
            GridWorld gridWorld = (GridWorld) environment;
            int[] position = gridWorld.positionFromState(state);
            int row = position[0];
            int col = position[1];
            
            switch (action) {
                case GridWorld.ACTION_UP:
                    row = Math.max(0, row - 1);
                    break;
                case GridWorld.ACTION_DOWN:
                    row = Math.min(gridWorld.getHeight() - 1, row + 1);
                    break;
                case GridWorld.ACTION_LEFT:
                    col = Math.max(0, col - 1);
                    break;
                case GridWorld.ACTION_RIGHT:
                    col = Math.min(gridWorld.getWidth() - 1, col + 1);
                    break;
            }
            
            return gridWorld.stateFromPosition(row, col);
        }
        
        return state; // Default: no state change
    }
    
    /**
     * Get the computed values
     */
    public double[] getValues() {
        return values.clone();
    }
    
    /**
     * Get the computed policy
     */
    public int[] getPolicy() {
        return policy.clone();
    }
    
    /**
     * Check if the algorithm converged
     */
    public boolean hasConverged() {
        return converged;
    }
    
    /**
     * Get the number of iterations used
     */
    public int getIterationsUsed() {
        return iterationsUsed;
    }
    
    /**
     * Get the value of a specific state
     */
    public double getStateValue(int state) {
        return values[state];
    }
    
    /**
     * Get the optimal action for a specific state
     */
    public int getOptimalAction(int state) {
        return policy[state];
    }
    
    /**
     * Print the values and policy
     */
    public void printResults() {
        System.out.println("Value Iteration Results:");
        System.out.println("Converged: " + converged);
        System.out.println("Iterations used: " + iterationsUsed);
        System.out.println();
        
        System.out.println("State Values:");
        for (int state = 0; state < values.length; state++) {
            System.out.printf("State %2d: Value = %8.4f, Action = %s%n", 
                            state, values[state], actionToString(policy[state]));
        }
        System.out.println();
    }
    
    /**
     * Convert action to string for display
     */
    private String actionToString(int action) {
        switch (action) {
            case GridWorld.ACTION_UP: return "UP";
            case GridWorld.ACTION_DOWN: return "DOWN";
            case GridWorld.ACTION_LEFT: return "LEFT";
            case GridWorld.ACTION_RIGHT: return "RIGHT";
            case -1: return "TERMINAL";
            default: return "UNKNOWN";
        }
    }
}

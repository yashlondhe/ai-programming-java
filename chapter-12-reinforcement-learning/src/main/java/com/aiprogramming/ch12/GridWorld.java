package com.aiprogramming.ch12;

import java.util.*;

/**
 * Grid World environment - a classic reinforcement learning problem
 * The agent moves in a grid trying to reach a goal while avoiding obstacles
 */
public class GridWorld extends Environment {
    
    // Actions
    public static final int ACTION_UP = 0;
    public static final int ACTION_DOWN = 1;
    public static final int ACTION_LEFT = 2;
    public static final int ACTION_RIGHT = 3;
    
    private final int width;
    private final int height;
    private final int[][] grid;
    private final Map<Integer, Double> rewards;
    private final Set<Integer> terminalStates;
    private int agentRow;
    private int agentCol;
    private int currentState;
    private boolean done;
    
    // Grid cell types
    public static final int EMPTY = 0;
    public static final int WALL = 1;
    public static final int GOAL = 2;
    public static final int PIT = 3;
    
    public GridWorld(int width, int height) {
        this.width = width;
        this.height = height;
        this.grid = new int[height][width];
        this.rewards = new HashMap<>();
        this.terminalStates = new HashSet<>();
        this.done = false;
        
        // Initialize empty grid
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                grid[i][j] = EMPTY;
            }
        }
        
        // Set default rewards
        setDefaultRewards();
    }
    
    /**
     * Create a simple 4x4 grid world with goal and pit
     */
    public static GridWorld createSimpleGridWorld() {
        GridWorld world = new GridWorld(4, 4);
        
        // Set goal at bottom-right
        world.setCellType(3, 3, GOAL);
        world.setReward(world.stateFromPosition(3, 3), 1.0);
        world.addTerminalState(world.stateFromPosition(3, 3));
        
        // Set pit at (1,1)
        world.setCellType(1, 1, PIT);
        world.setReward(world.stateFromPosition(1, 1), -1.0);
        world.addTerminalState(world.stateFromPosition(1, 1));
        
        // Set some walls
        world.setCellType(1, 2, WALL);
        world.setCellType(2, 1, WALL);
        
        return world;
    }
    
    private void setDefaultRewards() {
        // Default reward for each step
        for (int state = 0; state < getStateSize(); state++) {
            rewards.put(state, -0.04); // Small negative reward for each step
        }
    }
    
    public void setCellType(int row, int col, int type) {
        if (isValidPosition(row, col)) {
            grid[row][col] = type;
        }
    }
    
    public void setReward(int state, double reward) {
        rewards.put(state, reward);
    }
    
    public void addTerminalState(int state) {
        terminalStates.add(state);
    }
    
    @Override
    public int reset() {
        // Reset agent to top-left corner
        agentRow = 0;
        agentCol = 0;
        
        // Find first empty cell if (0,0) is blocked
        while (grid[agentRow][agentCol] == WALL) {
            agentCol++;
            if (agentCol >= width) {
                agentCol = 0;
                agentRow++;
            }
        }
        
        currentState = stateFromPosition(agentRow, agentCol);
        done = false;
        return currentState;
    }
    
    @Override
    public StepResult step(int action) {
        if (done) {
            throw new IllegalStateException("Episode is finished. Call reset() to start new episode.");
        }
        
        int newRow = agentRow;
        int newCol = agentCol;
        
        // Apply action
        switch (action) {
            case ACTION_UP:
                newRow = Math.max(0, agentRow - 1);
                break;
            case ACTION_DOWN:
                newRow = Math.min(height - 1, agentRow + 1);
                break;
            case ACTION_LEFT:
                newCol = Math.max(0, agentCol - 1);
                break;
            case ACTION_RIGHT:
                newCol = Math.min(width - 1, agentCol + 1);
                break;
            default:
                throw new IllegalArgumentException("Invalid action: " + action);
        }
        
        // Check if new position is valid (not a wall)
        if (grid[newRow][newCol] == WALL) {
            newRow = agentRow;
            newCol = agentCol;
        }
        
        // Update agent position
        agentRow = newRow;
        agentCol = newCol;
        currentState = stateFromPosition(agentRow, agentCol);
        
        // Get reward
        double reward = rewards.getOrDefault(currentState, 0.0);
        
        // Check if episode is done
        done = terminalStates.contains(currentState);
        
        return new StepResult(currentState, reward, done);
    }
    
    @Override
    public int getCurrentState() {
        return currentState;
    }
    
    @Override
    public int getStateSize() {
        return width * height;
    }
    
    @Override
    public int getActionSize() {
        return 4; // UP, DOWN, LEFT, RIGHT
    }
    
    @Override
    public boolean isDone() {
        return done;
    }
    
    @Override
    public void render() {
        System.out.println("\nGrid World:");
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (i == agentRow && j == agentCol) {
                    System.out.print("A ");
                } else {
                    switch (grid[i][j]) {
                        case EMPTY:
                            System.out.print(". ");
                            break;
                        case WALL:
                            System.out.print("# ");
                            break;
                        case GOAL:
                            System.out.print("G ");
                            break;
                        case PIT:
                            System.out.print("P ");
                            break;
                        default:
                            System.out.print("? ");
                    }
                }
            }
            System.out.println();
        }
        System.out.println();
    }
    
    @Override
    public boolean isTerminalState(int state) {
        return terminalStates.contains(state);
    }
    
    @Override
    public int[] getPossibleActions(int state) {
        // In this simple grid world, all actions are always possible
        // The environment will handle invalid moves
        return new int[]{ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT};
    }
    
    /**
     * Convert (row, col) position to state index
     */
    public int stateFromPosition(int row, int col) {
        return row * width + col;
    }
    
    /**
     * Convert state index to (row, col) position
     */
    public int[] positionFromState(int state) {
        int row = state / width;
        int col = state % width;
        return new int[]{row, col};
    }
    
    /**
     * Check if position is within bounds
     */
    private boolean isValidPosition(int row, int col) {
        return row >= 0 && row < height && col >= 0 && col < width;
    }
    
    /**
     * Get reward for a specific state
     */
    public double getReward(int state) {
        return rewards.getOrDefault(state, 0.0);
    }
    
    /**
     * Get the grid dimensions
     */
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    /**
     * Get the agent's current position
     */
    public int[] getAgentPosition() {
        return new int[]{agentRow, agentCol};
    }
}

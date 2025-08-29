# Chapter 12: Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a branch of machine learning where an agent learns to make optimal decisions by interacting with an environment. Unlike supervised learning, RL agents don't have access to labeled training data. Instead, they learn through trial and error, receiving rewards or penalties based on their actions. This approach is inspired by how humans and animals learn through experience.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand the fundamental concepts of reinforcement learning and Markov Decision Processes
- Implement temporal difference learning algorithms (Q-Learning and SARSA)
- Apply dynamic programming methods (Value Iteration and Policy Iteration)
- Design and implement custom environments for RL experiments
- Train and evaluate RL agents using different algorithms
- Compare the performance of various RL approaches
- Understand the exploration vs exploitation trade-off

### Key Concepts

- **Agent**: The learner or decision maker that interacts with the environment
- **Environment**: The external system with which the agent interacts
- **State**: A representation of the current situation of the agent
- **Action**: A choice or decision the agent can make
- **Reward**: Feedback from the environment indicating the quality of an action
- **Policy**: A strategy or mapping from states to actions
- **Value Function**: Estimate of expected future rewards from a given state
- **Q-Function**: Estimate of expected future rewards for state-action pairs

## 12.1 Markov Decision Processes

Reinforcement learning problems are typically formulated as Markov Decision Processes (MDPs), which provide a mathematical framework for modeling decision-making situations.

### 12.1.1 MDP Components

An MDP is defined by:
- **States (S)**: Set of all possible states
- **Actions (A)**: Set of all possible actions
- **Transition Function (P)**: Probability of reaching state s' from state s taking action a
- **Reward Function (R)**: Expected reward for taking action a in state s
- **Discount Factor (γ)**: Factor that weights future rewards

#### Environment Interface

```java
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
```

### 12.1.2 Grid World Environment

The Grid World is a classic RL environment where an agent navigates a grid to reach a goal while avoiding obstacles.

#### Implementation

```java
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
    
    // Additional helper methods...
}
```

## 12.2 Temporal Difference Learning

Temporal Difference (TD) learning is a fundamental concept in reinforcement learning that combines ideas from Monte Carlo methods and dynamic programming.

### 12.2.1 Q-Learning

Q-Learning is an off-policy temporal difference learning algorithm that learns the optimal action-value function.

#### Implementation

```java
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
    
    // Additional methods for Q-table analysis...
}
```

#### Key Features

- **Off-policy Learning**: Learns optimal policy while following exploratory policy
- **Epsilon-greedy Exploration**: Balances exploration and exploitation
- **Q-table**: Stores action-value estimates for all state-action pairs
- **Temporal Difference Updates**: Updates estimates based on observed rewards

### 12.2.2 SARSA

SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning algorithm.

#### Implementation

```java
package com.aiprogramming.ch12;

/**
 * SARSA (State-Action-Reward-State-Action) agent implementation
 * SARSA is an on-policy reinforcement learning algorithm
 */
public class SarsaAgent extends Agent {
    
    private final double[][] qTable;
    private double explorationRate;
    private final Random random;
    
    public SarsaAgent(int stateSize, int actionSize, double learningRate, 
                     double discountFactor, double explorationRate) {
        super(stateSize, actionSize, learningRate, discountFactor);
        this.qTable = new double[stateSize][actionSize];
        this.explorationRate = explorationRate;
        this.random = new Random();
        
        initializeQTable();
    }
    
    /**
     * SARSA update with explicit next action
     * This is the proper SARSA update rule
     */
    public void learnSarsa(int state, int action, double reward, 
                          int nextState, int nextAction, boolean done) {
        // SARSA update rule:
        // Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        
        double currentQValue = qTable[state][action];
        double nextQValue = done ? 0.0 : qTable[nextState][nextAction];
        
        double targetValue = reward + discountFactor * nextQValue;
        double tdError = targetValue - currentQValue;
        
        qTable[state][action] = currentQValue + learningRate * tdError;
    }
    
    // Additional SARSA-specific methods...
}
```

#### Differences from Q-Learning

- **On-policy**: Learns about the policy being executed
- **Action Selection**: Uses the action that will actually be taken next
- **Conservative**: More conservative in dangerous environments
- **Exploration Aware**: Takes exploration into account during learning

## 12.3 Dynamic Programming

Dynamic programming methods solve MDPs when the model (transition probabilities and rewards) is known.

### 12.3.1 Value Iteration

Value Iteration computes optimal values by iteratively applying the Bellman optimality equation.

#### Implementation

```java
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
    
    // Additional helper methods...
}
```

#### Algorithm Steps

1. **Initialize**: Set initial value estimates (usually zeros)
2. **Update**: For each state, compute optimal value using Bellman equation
3. **Check Convergence**: Stop when value changes are below tolerance
4. **Extract Policy**: Derive optimal policy from computed values

### 12.3.2 Policy Iteration

Policy Iteration alternates between policy evaluation and policy improvement.

#### Implementation

```java
package com.aiprogramming.ch12;

/**
 * Policy Iteration algorithm for solving Markov Decision Processes (MDPs)
 * Alternates between policy evaluation and policy improvement
 */
public class PolicyIteration {
    
    private final Environment environment;
    private double[] values;
    private int[] policy;
    
    /**
     * Run policy iteration algorithm
     */
    public void solve() {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Policy Evaluation
            evaluatePolicy();
            
            // Policy Improvement
            boolean policyChanged = improvePolicy();
            
            iterationsUsed = iteration + 1;
            
            // Check for convergence
            if (!policyChanged) {
                converged = true;
                break;
            }
        }
    }
    
    /**
     * Evaluate the current policy by computing state values
     */
    private void evaluatePolicy() {
        // Iteratively compute values until convergence
        for (int iter = 0; iter < 100; iter++) {
            double maxChange = 0.0;
            
            for (int state = 0; state < stateSize; state++) {
                if (environment.isTerminalState(state)) {
                    newValues[state] = getRewardForState(state);
                } else {
                    int action = policy[state];
                    newValues[state] = computeActionValue(state, action);
                }
                
                double change = Math.abs(newValues[state] - values[state]);
                maxChange = Math.max(maxChange, change);
            }
            
            values = newValues.clone();
            
            if (maxChange < tolerance) {
                break;
            }
        }
    }
    
    /**
     * Improve the policy by choosing the best action for each state
     */
    private boolean improvePolicy() {
        boolean policyChanged = false;
        
        for (int state = 0; state < stateSize; state++) {
            if (environment.isTerminalState(state)) {
                continue;
            }
            
            int oldAction = policy[state];
            int bestAction = findBestAction(state);
            
            policy[state] = bestAction;
            
            if (oldAction != bestAction) {
                policyChanged = true;
            }
        }
        
        return policyChanged;
    }
    
    // Additional methods...
}
```

## 12.4 Training and Evaluation

### 12.4.1 Training Framework

A comprehensive training framework provides utilities for agent training and performance evaluation.

#### Implementation

```java
package com.aiprogramming.ch12;

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
            
            while (!environment.isDone() && stepCount < 1000) {
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
        // Set exploration to zero for evaluation
        double originalExplorationRate = agent.getExplorationRate();
        agent.setExplorationRate(0.0);
        
        List<Double> episodeRewards = new ArrayList<>();
        List<Integer> episodeLengths = new ArrayList<>();
        
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
    
    // Additional training utilities...
}
```

### 12.4.2 Performance Metrics

Key metrics for evaluating RL agents:

- **Average Reward**: Mean reward per episode
- **Success Rate**: Percentage of episodes reaching the goal
- **Episode Length**: Average number of steps to completion
- **Learning Curve**: Reward progression over training episodes
- **Convergence**: Stability of performance metrics

## 12.5 Practical Example

### 12.5.1 Grid World Navigation

A comprehensive example demonstrating all implemented algorithms:

```java
package com.aiprogramming.ch12;

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
        
        // Evaluate performance
        qResult.printSummary();
        
        // Show learned policy
        qAgent.printQTable();
        ReinforcementLearningTrainer.demonstrateAgent(qAgent, environment, true, true);
    }
    
    // Additional demonstration methods...
}
```

### 12.5.2 Expected Results

The example typically produces:

1. **Q-Learning**: Converges to optimal policy, high success rate
2. **SARSA**: Similar performance but more conservative behavior
3. **Value Iteration**: Finds optimal policy in few iterations
4. **Policy Iteration**: Alternative optimal solution method

## 12.6 Advanced Topics

### 12.6.1 Exploration Strategies

Beyond epsilon-greedy:
- **Upper Confidence Bounds (UCB)**: Optimistic action selection
- **Thompson Sampling**: Bayesian approach to exploration
- **Entropy-based Methods**: Information-theoretic exploration

### 12.6.2 Function Approximation

For large state spaces:
- **Linear Function Approximation**: Feature-based value functions
- **Neural Networks**: Deep Q-Learning (DQN)
- **Tile Coding**: Discrete approximation methods

### 12.6.3 Policy Gradient Methods

Direct policy optimization:
- **REINFORCE**: Basic policy gradient algorithm
- **Actor-Critic**: Combines value functions with policy gradients
- **Proximal Policy Optimization (PPO)**: State-of-the-art policy method

## 12.7 Best Practices

### 12.7.1 Hyperparameter Tuning

- **Learning Rate**: Start with 0.1, adjust based on convergence
- **Discount Factor**: Use 0.95-0.99 for most problems
- **Exploration Rate**: Start high (0.9), decay to low (0.01)
- **Episode Length**: Balance exploration vs computational cost

### 12.7.2 Environment Design

- **State Representation**: Clear, complete, and minimal
- **Reward Structure**: Aligned with desired behavior
- **Terminal Conditions**: Well-defined episode endings
- **Action Space**: Appropriate granularity for the problem

### 12.7.3 Training Strategy

- **Sufficient Episodes**: Train until convergence
- **Evaluation Protocol**: Separate training and testing
- **Multiple Runs**: Average results over multiple random seeds
- **Baseline Comparison**: Compare against simple heuristics

## 12.8 Summary

In this chapter, we explored reinforcement learning fundamentals:

1. **Markov Decision Processes**: Mathematical framework for RL problems
2. **Temporal Difference Learning**: Q-Learning and SARSA algorithms
3. **Dynamic Programming**: Value Iteration and Policy Iteration methods
4. **Environment Design**: Grid World implementation and customization
5. **Training Framework**: Comprehensive agent training and evaluation
6. **Performance Analysis**: Metrics and comparison methodologies

### Key Takeaways

- **Reinforcement Learning** enables agents to learn optimal behavior through interaction
- **Exploration vs Exploitation** is a fundamental trade-off in RL
- **Temporal Difference Learning** provides efficient online learning algorithms
- **Dynamic Programming** offers exact solutions for known MDPs
- **Proper Evaluation** requires careful experimental design and metrics

### Next Steps

- Explore deep reinforcement learning with neural networks
- Implement more sophisticated exploration strategies
- Apply RL to continuous state and action spaces
- Study multi-agent reinforcement learning
- Investigate real-world RL applications

## Exercises

### Exercise 1: Custom Environment
Create a new environment (e.g., maze, cart-pole) and test different RL algorithms on it.

### Exercise 2: Algorithm Comparison
Implement additional algorithms (Double Q-Learning, Expected SARSA) and compare their performance.

### Exercise 3: Hyperparameter Study
Systematically study the effect of different hyperparameters on learning performance.

### Exercise 4: Policy Analysis
Analyze learned policies and value functions to understand agent behavior.

### Exercise 5: Exploration Strategies
Implement and compare different exploration strategies beyond epsilon-greedy.

### Exercise 6: Function Approximation
Extend the framework to handle larger state spaces using function approximation.

### Exercise 7: Multi-Agent RL
Modify the environment to support multiple agents and study their interactions.

### Exercise 8: Real-World Application
Apply the RL framework to a practical problem domain of your choice.

package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;

/**
 * Optimizer interface
 */
public interface Optimizer {
    double getLearningRate();
    void updateLearningRate(double newLearningRate);
}

/**
 * Stochastic Gradient Descent (SGD)
 */
class SGD implements Optimizer {
    private double learningRate;
    private double momentum;
    private List<Double> velocity;
    
    public SGD(double learningRate) {
        this(learningRate, 0.0);
    }
    
    public SGD(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocity = new ArrayList<>();
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
    
    @Override
    public void updateLearningRate(double newLearningRate) {
        this.learningRate = newLearningRate;
    }
    
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
}

/**
 * Adam optimizer
 */
class Adam implements Optimizer {
    private double learningRate;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int t;
    private List<Double> m; // First moment
    private List<Double> v; // Second moment
    
    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }
    
    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.m = new ArrayList<>();
        this.v = new ArrayList<>();
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
    
    @Override
    public void updateLearningRate(double newLearningRate) {
        this.learningRate = newLearningRate;
    }
}

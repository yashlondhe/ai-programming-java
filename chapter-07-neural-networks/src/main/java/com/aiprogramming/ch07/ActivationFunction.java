package com.aiprogramming.ch07;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Activation function interface and implementations
 */
public interface ActivationFunction {
    double apply(double x);
    double derivative(double x);
}

/**
 * Sigmoid activation function: 1 / (1 + e^(-x))
 */
class Sigmoid implements ActivationFunction {
    @Override
    public double apply(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    @Override
    public double derivative(double x) {
        double sigmoid = apply(x);
        return sigmoid * (1.0 - sigmoid);
    }
}

/**
 * ReLU (Rectified Linear Unit) activation function: max(0, x)
 */
class ReLU implements ActivationFunction {
    @Override
    public double apply(double x) {
        return Math.max(0, x);
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}

/**
 * Tanh (Hyperbolic Tangent) activation function: (e^x - e^(-x)) / (e^x + e^(-x))
 */
class Tanh implements ActivationFunction {
    @Override
    public double apply(double x) {
        return Math.tanh(x);
    }
    
    @Override
    public double derivative(double x) {
        double tanh = apply(x);
        return 1.0 - tanh * tanh;
    }
}

/**
 * Leaky ReLU activation function: max(0.01x, x)
 */
class LeakyReLU implements ActivationFunction {
    private final double alpha;
    
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }
    
    public LeakyReLU() {
        this(0.01);
    }
    
    @Override
    public double apply(double x) {
        return x > 0 ? x : alpha * x;
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : alpha;
    }
}

/**
 * Softmax activation function for output layers
 */
class Softmax implements ActivationFunction {
    private List<Double> lastOutputs;
    
    @Override
    public double apply(double x) {
        // Softmax is applied to a vector, not a single value
        // This is a placeholder - actual implementation is in the layer
        return x;
    }
    
    @Override
    public double derivative(double x) {
        // Derivative is computed differently for softmax
        return x;
    }
    
    public List<Double> applyToVector(List<Double> inputs) {
        // Compute softmax: exp(x_i) / sum(exp(x_j))
        double maxInput = inputs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        
        List<Double> expInputs = inputs.stream()
                .map(input -> Math.exp(input - maxInput)) // Subtract max for numerical stability
                .collect(Collectors.toList());
        
        double sumExp = expInputs.stream().mapToDouble(Double::doubleValue).sum();
        
        this.lastOutputs = expInputs.stream()
                .map(exp -> exp / sumExp)
                .collect(Collectors.toList());
        
        return new ArrayList<>(this.lastOutputs);
    }
    
    public List<Double> derivativeForVector(List<Double> targets) {
        // Compute softmax derivative: softmax_i * (target_i - softmax_i)
        List<Double> derivatives = new ArrayList<>();
        for (int i = 0; i < lastOutputs.size(); i++) {
            derivatives.add(lastOutputs.get(i) * (targets.get(i) - lastOutputs.get(i)));
        }
        return derivatives;
    }
}

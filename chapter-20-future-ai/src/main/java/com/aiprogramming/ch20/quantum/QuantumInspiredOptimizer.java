package com.aiprogramming.ch20.quantum;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;

/**
 * Quantum-Inspired Optimization Algorithm
 * 
 * This implementation simulates quantum computing concepts like superposition,
 * entanglement, and quantum tunneling to solve optimization problems.
 * It's inspired by quantum annealing and quantum particle swarm optimization.
 */
public class QuantumInspiredOptimizer {
    
    private static final Logger logger = LoggerFactory.getLogger(QuantumInspiredOptimizer.class);
    
    private final int populationSize;
    private final int maxIterations;
    private final double quantumRadius;
    private final double tunnelingProbability;
    private final RandomGenerator random;
    
    public QuantumInspiredOptimizer() {
        this(50, 1000, 0.1, 0.05);
    }
    
    public QuantumInspiredOptimizer(int populationSize, int maxIterations, 
                                   double quantumRadius, double tunnelingProbability) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.quantumRadius = quantumRadius;
        this.tunnelingProbability = tunnelingProbability;
        this.random = new Well19937c();
    }
    
    /**
     * Represents a quantum particle with position and velocity
     */
    public static class QuantumParticle {
        private RealVector position;
        private RealVector velocity;
        private RealVector bestPosition;
        private double bestFitness;
        private double quantumState; // Represents superposition state
        
        public QuantumParticle(int dimension, RandomGenerator random) {
            this.position = new ArrayRealVector(dimension);
            this.velocity = new ArrayRealVector(dimension);
            this.bestPosition = new ArrayRealVector(dimension);
            this.quantumState = random.nextDouble() * 2 * Math.PI;
        }
        
        public void initialize(RandomGenerator random, double[] bounds) {
            for (int i = 0; i < position.getDimension(); i++) {
                double min = bounds[i * 2];
                double max = bounds[i * 2 + 1];
                position.setEntry(i, min + random.nextDouble() * (max - min));
                velocity.setEntry(i, (random.nextDouble() - 0.5) * 0.1);
            }
            bestPosition = position.copy();
            bestFitness = Double.POSITIVE_INFINITY;
        }
    }
    
    /**
     * Optimize a function using quantum-inspired algorithm
     * 
     * @param fitnessFunction The function to optimize
     * @param bounds Array of [min, max] bounds for each dimension
     * @return Best solution found
     */
    public double[] optimize(OptimizationFunction fitnessFunction, double[] bounds) {
        logger.info("Starting quantum-inspired optimization with {} particles", populationSize);
        
        // Initialize quantum particles
        List<QuantumParticle> particles = new ArrayList<>();
        int dimension = bounds.length / 2;
        
        for (int i = 0; i < populationSize; i++) {
            QuantumParticle particle = new QuantumParticle(dimension, random);
            particle.initialize(random, bounds);
            particles.add(particle);
        }
        
        // Find global best
        QuantumParticle globalBest = particles.get(0);
        for (QuantumParticle particle : particles) {
            double fitness = fitnessFunction.evaluate(particle.position.toArray());
            particle.bestFitness = fitness;
            if (fitness < globalBest.bestFitness) {
                globalBest = particle;
            }
        }
        
        // Main optimization loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            
            // Update quantum states and apply quantum effects
            for (QuantumParticle particle : particles) {
                updateQuantumState(particle);
                applyQuantumTunneling(particle, globalBest);
                updatePosition(particle, globalBest, iteration);
                
                // Evaluate fitness
                double fitness = fitnessFunction.evaluate(particle.position.toArray());
                
                // Update personal best
                if (fitness < particle.bestFitness) {
                    particle.bestPosition = particle.position.copy();
                    particle.bestFitness = fitness;
                    
                    // Update global best
                    if (fitness < globalBest.bestFitness) {
                        globalBest = particle;
                        logger.debug("New global best found at iteration {}: {}", iteration, fitness);
                    }
                }
            }
            
            // Apply quantum entanglement between particles
            applyQuantumEntanglement(particles);
            
            if (iteration % 100 == 0) {
                logger.info("Iteration {}: Best fitness = {}", iteration, globalBest.bestFitness);
            }
        }
        
        logger.info("Optimization completed. Best fitness: {}", globalBest.bestFitness);
        return globalBest.bestPosition.toArray();
    }
    
    /**
     * Update quantum state of a particle (simulates superposition)
     */
    private void updateQuantumState(QuantumParticle particle) {
        // Simulate quantum superposition by adding quantum fluctuations
        double quantumFluctuation = Math.sin(particle.quantumState) * quantumRadius;
        particle.quantumState += random.nextDouble() * 0.1;
        
        // Apply quantum fluctuations to position
        for (int i = 0; i < particle.position.getDimension(); i++) {
            double currentPos = particle.position.getEntry(i);
            particle.position.setEntry(i, currentPos + quantumFluctuation * (random.nextDouble() - 0.5));
        }
    }
    
    /**
     * Apply quantum tunneling effect
     */
    private void applyQuantumTunneling(QuantumParticle particle, QuantumParticle globalBest) {
        if (random.nextDouble() < tunnelingProbability) {
            // Simulate quantum tunneling by allowing particles to "tunnel" through barriers
            double tunnelingDistance = quantumRadius * Math.log(1.0 / random.nextDouble());
            
            for (int i = 0; i < particle.position.getDimension(); i++) {
                double direction = random.nextDouble() - 0.5;
                double newPos = particle.position.getEntry(i) + tunnelingDistance * direction;
                particle.position.setEntry(i, newPos);
            }
        }
    }
    
    /**
     * Update particle position using quantum-inspired velocity update
     */
    private void updatePosition(QuantumParticle particle, QuantumParticle globalBest, int iteration) {
        double w = 0.7; // Inertia weight
        double c1 = 1.5; // Cognitive coefficient
        double c2 = 1.5; // Social coefficient
        
        for (int i = 0; i < particle.position.getDimension(); i++) {
            // Quantum-inspired velocity update
            double cognitive = c1 * random.nextDouble() * (particle.bestPosition.getEntry(i) - particle.position.getEntry(i));
            double social = c2 * random.nextDouble() * (globalBest.bestPosition.getEntry(i) - particle.position.getEntry(i));
            
            // Add quantum effects to velocity
            double quantumEffect = Math.sin(particle.quantumState) * quantumRadius * 0.1;
            
            particle.velocity.setEntry(i, w * particle.velocity.getEntry(i) + cognitive + social + quantumEffect);
            particle.position.setEntry(i, particle.position.getEntry(i) + particle.velocity.getEntry(i));
        }
    }
    
    /**
     * Apply quantum entanglement between particles
     */
    private void applyQuantumEntanglement(List<QuantumParticle> particles) {
        // Simulate quantum entanglement by correlating particle movements
        for (int i = 0; i < particles.size() - 1; i += 2) {
            QuantumParticle p1 = particles.get(i);
            QuantumParticle p2 = particles.get(i + 1);
            
            // Create entanglement effect
            double entanglementStrength = 0.1;
            for (int j = 0; j < p1.position.getDimension(); j++) {
                double avgPos = (p1.position.getEntry(j) + p2.position.getEntry(j)) / 2.0;
                p1.position.setEntry(j, avgPos + entanglementStrength * (random.nextDouble() - 0.5));
                p2.position.setEntry(j, avgPos + entanglementStrength * (random.nextDouble() - 0.5));
            }
        }
    }
    
    /**
     * Interface for optimization functions
     */
    @FunctionalInterface
    public interface OptimizationFunction {
        double evaluate(double[] position);
    }
    
    /**
     * Example optimization functions
     */
    public static class OptimizationFunctions {
        
        /**
         * Sphere function (minimum at origin)
         */
        public static OptimizationFunction sphere() {
            return position -> {
                double sum = 0;
                for (double x : position) {
                    sum += x * x;
                }
                return sum;
            };
        }
        
        /**
         * Rastrigin function (many local minima)
         */
        public static OptimizationFunction rastrigin() {
            return position -> {
                double sum = 0;
                for (double x : position) {
                    sum += x * x - 10 * Math.cos(2 * Math.PI * x) + 10;
                }
                return sum;
            };
        }
        
        /**
         * Rosenbrock function (valley-shaped)
         */
        public static OptimizationFunction rosenbrock() {
            return position -> {
                double sum = 0;
                for (int i = 0; i < position.length - 1; i++) {
                    double x = position[i];
                    double y = position[i + 1];
                    sum += 100 * Math.pow(y - x * x, 2) + Math.pow(1 - x, 2);
                }
                return sum;
            };
        }
    }
}

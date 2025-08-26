package com.aiprogramming.ch05;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Unit tests for Regression Evaluator
 */
public class RegressionEvaluatorTest {
    
    private RegressionEvaluator evaluator;
    
    @BeforeEach
    void setUp() {
        evaluator = new RegressionEvaluator();
    }
    
    @Test
    void testPerfectPredictions() {
        List<Double> actual = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0);
        List<Double> predicted = Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0);
        
        RegressionMetrics metrics = evaluator.evaluate(actual, predicted);
        
        assertEquals(0.0, metrics.getMAE(), 1e-10);
        assertEquals(0.0, metrics.getMSE(), 1e-10);
        assertEquals(0.0, metrics.getRMSE(), 1e-10);
        assertEquals(1.0, metrics.getR2(), 1e-10);
        assertEquals(0.0, metrics.getMAPE(), 1e-10);
    }
    
    @Test
    void testKnownMetrics() {
        List<Double> actual = Arrays.asList(3.0, -0.5, 2.0, 7.0);
        List<Double> predicted = Arrays.asList(2.5, 0.0, 2.0, 8.0);
        
        RegressionMetrics metrics = evaluator.evaluate(actual, predicted);
        
        // MAE = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
        assertEquals(0.5, metrics.getMAE(), 1e-10);
        
        // MSE = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
        assertEquals(0.375, metrics.getMSE(), 1e-10);
        
        // RMSE = sqrt(0.375) ≈ 0.6124
        assertEquals(Math.sqrt(0.375), metrics.getRMSE(), 1e-10);
        
        // Residuals should be [0.5, -0.5, 0.0, -1.0]
        List<Double> expectedResiduals = Arrays.asList(0.5, -0.5, 0.0, -1.0);
        assertEquals(expectedResiduals, metrics.getResiduals());
    }
    
    @Test
    void testR2Calculation() {
        // Test case where R² = 0.5
        List<Double> actual = Arrays.asList(1.0, 2.0, 3.0, 4.0);
        List<Double> predicted = Arrays.asList(1.5, 2.5, 2.5, 3.5);
        
        RegressionMetrics metrics = evaluator.evaluate(actual, predicted);
        
        // Manual calculation:
        // Mean = 2.5
        // TSS = (1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)² = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        // RSS = (1-1.5)² + (2-2.5)² + (3-2.5)² + (4-3.5)² = 0.25 + 0.25 + 0.25 + 0.25 = 1.0
        // R² = 1 - RSS/TSS = 1 - 1.0/5.0 = 0.8
        assertEquals(0.8, metrics.getR2(), 1e-10);
    }
    
    @Test
    void testMAPEWithZeroValues() {
        List<Double> actual = Arrays.asList(0.0, 2.0, 4.0);
        List<Double> predicted = Arrays.asList(1.0, 2.0, 3.0);
        
        RegressionMetrics metrics = evaluator.evaluate(actual, predicted);
        
        // MAPE should skip zero values
        // Only calculate for non-zero actual values: |2-2|/2 + |4-3|/4 = 0 + 0.25 = 0.25
        // Average: 0.25 / 2 = 0.125 = 12.5%
        assertEquals(12.5, metrics.getMAPE(), 1e-10);
    }
    
    @Test
    void testEmptyLists() {
        assertThrows(IllegalArgumentException.class, () -> {
            evaluator.evaluate(new ArrayList<>(), new ArrayList<>());
        });
    }
    
    @Test
    void testMismatchedSizes() {
        List<Double> actual = Arrays.asList(1.0, 2.0, 3.0);
        List<Double> predicted = Arrays.asList(1.0, 2.0);
        
        assertThrows(IllegalArgumentException.class, () -> {
            evaluator.evaluate(actual, predicted);
        });
    }
    
    @Test
    void testConstantActualValues() {
        // When all actual values are the same, R² should be 0 or undefined
        List<Double> actual = Arrays.asList(5.0, 5.0, 5.0, 5.0);
        List<Double> predicted = Arrays.asList(4.0, 5.0, 6.0, 5.0);
        
        RegressionMetrics metrics = evaluator.evaluate(actual, predicted);
        
        // TSS = 0, so R² should be 0 (handled specially)
        assertEquals(0.0, metrics.getR2(), 1e-10);
    }
}
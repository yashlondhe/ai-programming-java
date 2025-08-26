package com.aiprogramming.ch05;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Unit tests for Linear Regression implementation
 */
public class LinearRegressionTest {
    
    private LinearRegression regression;
    private List<RegressionDataPoint> simpleData;
    private List<RegressionDataPoint> multipleData;
    
    @BeforeEach
    void setUp() {
        regression = new LinearRegression();
        
        // Simple linear data: y = 2x + 1
        simpleData = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("x", (double) i);
            double target = 2.0 * i + 1.0;
            simpleData.add(new RegressionDataPoint(features, target));
        }
        
        // Multiple features data: y = 2x1 + 3x2 + 1 (with less correlation)
        multipleData = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            Map<String, Double> features = new HashMap<>();
            features.put("x1", (double) i);
            features.put("x2", (double) (i + 2)); // Less correlated with x1
            double target = 2.0 * i + 3.0 * (i + 2) + 1.0;
            multipleData.add(new RegressionDataPoint(features, target));
        }
    }
    
    @Test
    void testTrainWithEmptyData() {
        assertThrows(IllegalArgumentException.class, () -> {
            regression.train(new ArrayList<>());
        });
    }
    
    @Test
    void testPredictWithoutTraining() {
        Map<String, Double> features = new HashMap<>();
        features.put("x", 1.0);
        
        assertThrows(IllegalStateException.class, () -> {
            regression.predict(features);
        });
    }
    
    @Test
    void testSimpleLinearRegression() {
        regression.train(simpleData);
        
        // Test prediction
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("x", 6.0);
        double prediction = regression.predict(testFeatures);
        
        // Should predict y = 2*6 + 1 = 13
        assertEquals(13.0, prediction, 1e-10);
        
        // Test coefficients
        Map<String, Double> coefficients = regression.getCoefficients();
        assertEquals(2.0, coefficients.get("x"), 1e-10);
        assertEquals(1.0, regression.getIntercept(), 1e-10);
    }
    
    @Test
    void testMultipleLinearRegression() {
        regression.train(multipleData);
        
        // Test prediction
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("x1", 11.0);
        testFeatures.put("x2", 13.0); // x2 = x1 + 2
        double prediction = regression.predict(testFeatures);
        
        // Should predict y = 2*11 + 3*13 + 1 = 22 + 39 + 1 = 62
        assertEquals(62.0, prediction, 1e-10);
        
        // Test coefficients
        Map<String, Double> coefficients = regression.getCoefficients();
        assertEquals(2.0, coefficients.get("x1"), 1e-10);
        assertEquals(3.0, coefficients.get("x2"), 1e-10);
        assertEquals(1.0, regression.getIntercept(), 1e-10);
    }
    
    @Test
    void testFeatureImportance() {
        regression.train(multipleData);
        
        Map<String, Double> importance = regression.getFeatureImportance();
        
        // x2 coefficient (3.0) should have higher importance than x1 (2.0)
        assertTrue(importance.get("x2") > importance.get("x1"));
        
        // Importance should be normalized
        assertTrue(importance.values().stream().allMatch(v -> v >= 0.0 && v <= 1.0));
    }
    
    @Test
    void testMissingFeatures() {
        regression.train(simpleData);
        
        // Test with missing feature (should default to 0)
        Map<String, Double> testFeatures = new HashMap<>();
        double prediction = regression.predict(testFeatures);
        
        // Should predict intercept only
        assertEquals(1.0, prediction, 1e-10);
    }
    
    @Test
    void testExtraFeatures() {
        regression.train(simpleData);
        
        // Test with extra feature (should be ignored)
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("x", 3.0);
        testFeatures.put("extra", 100.0);
        double prediction = regression.predict(testFeatures);
        
        // Should predict y = 2*3 + 1 = 7
        assertEquals(7.0, prediction, 1e-10);
    }
}
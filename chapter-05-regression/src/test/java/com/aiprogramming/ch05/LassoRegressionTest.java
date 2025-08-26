package com.aiprogramming.ch05;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Unit tests for Lasso Regression implementation
 */
public class LassoRegressionTest {
    
    private LassoRegression lasso;
    private List<RegressionDataPoint> sparseData;
    
    @BeforeEach
    void setUp() {
        // Create data where only some features are relevant
        // y = 2*x1 + 0*x2 + 3*x3 + noise
        sparseData = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 100; i++) {
            Map<String, Double> features = new HashMap<>();
            double x1 = random.nextGaussian();
            double x2 = random.nextGaussian(); // irrelevant feature
            double x3 = random.nextGaussian();
            
            features.put("x1", x1);
            features.put("x2", x2);
            features.put("x3", x3);
            
            double target = 2.0 * x1 + 3.0 * x3 + random.nextGaussian() * 0.1;
            sparseData.add(new RegressionDataPoint(features, target));
        }
    }
    
    @Test
    void testTrainWithValidData() {
        lasso = new LassoRegression(0.1);
        
        assertDoesNotThrow(() -> {
            lasso.train(sparseData);
        });
    }
    
    @Test
    void testInvalidAlpha() {
        assertThrows(IllegalArgumentException.class, () -> {
            new LassoRegression(-0.1);
        });
    }
    
    @Test
    void testFeatureSelection() {
        lasso = new LassoRegression(0.5); // High regularization
        lasso.train(sparseData);
        
        Set<String> selectedFeatures = lasso.getSelectedFeatures();
        
        // With high regularization, some features should be selected
        assertTrue(selectedFeatures.size() <= sparseData.get(0).getFeatureNames().size());
        
        // x1 and x3 should be more likely to be selected than x2
        Map<String, Double> coefficients = lasso.getCoefficients();
        double x2Coeff = Math.abs(coefficients.getOrDefault("x2", 0.0));
        double x1Coeff = Math.abs(coefficients.getOrDefault("x1", 0.0));
        double x3Coeff = Math.abs(coefficients.getOrDefault("x3", 0.0));
        
        // x2 should have smaller coefficient (it's irrelevant)
        assertTrue(x2Coeff <= Math.max(x1Coeff, x3Coeff));
    }
    
    @Test
    void testSparsityIncreaseWithAlpha() {
        LassoRegression lowAlpha = new LassoRegression(0.01);
        LassoRegression highAlpha = new LassoRegression(1.0);
        
        lowAlpha.train(sparseData);
        highAlpha.train(sparseData);
        
        // Higher alpha should lead to more sparsity
        assertTrue(highAlpha.getSparsity() >= lowAlpha.getSparsity());
    }
    
    @Test
    void testPrediction() {
        lasso = new LassoRegression(0.1);
        lasso.train(sparseData);
        
        Map<String, Double> testFeatures = new HashMap<>();
        testFeatures.put("x1", 1.0);
        testFeatures.put("x2", 1.0);
        testFeatures.put("x3", 1.0);
        
        double prediction = lasso.predict(testFeatures);
        
        // Should return a reasonable prediction
        assertFalse(Double.isNaN(prediction));
        assertFalse(Double.isInfinite(prediction));
    }
    
    @Test
    void testZeroAlpha() {
        // With alpha = 0, Lasso should behave similar to linear regression
        lasso = new LassoRegression(0.0);
        lasso.train(sparseData);
        
        // All features should be selected (no regularization)
        Set<String> selectedFeatures = lasso.getSelectedFeatures();
        assertEquals(sparseData.get(0).getFeatureNames().size(), selectedFeatures.size());
        
        // Sparsity should be low
        assertTrue(lasso.getSparsity() < 0.5);
    }
    
    @Test
    void testGetters() {
        lasso = new LassoRegression(0.5);
        
        assertEquals(0.5, lasso.getAlpha(), 1e-10);
        
        lasso.train(sparseData);
        
        assertNotNull(lasso.getCoefficients());
        assertFalse(Double.isNaN(lasso.getIntercept()));
        assertTrue(lasso.getSparsity() >= 0.0 && lasso.getSparsity() <= 1.0);
    }
    
    @Test
    void testEmptyData() {
        lasso = new LassoRegression(0.1);
        
        assertThrows(IllegalArgumentException.class, () -> {
            lasso.train(new ArrayList<>());
        });
    }
    
    @Test
    void testPredictWithoutTraining() {
        lasso = new LassoRegression(0.1);
        
        Map<String, Double> features = new HashMap<>();
        features.put("x1", 1.0);
        
        assertThrows(IllegalStateException.class, () -> {
            lasso.predict(features);
        });
    }
}
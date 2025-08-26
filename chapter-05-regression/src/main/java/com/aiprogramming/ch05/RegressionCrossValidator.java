package com.aiprogramming.ch05;

import java.util.*;

/**
 * K-Fold Cross-Validation for regression
 */
public class RegressionCrossValidator {
    
    public static RegressionCrossValidationResults kFoldCrossValidation(
            Regressor regressor, List<RegressionDataPoint> data, int k) {
        
        List<Double> maes = new ArrayList<>();
        List<Double> mses = new ArrayList<>();
        List<Double> rmses = new ArrayList<>();
        List<Double> r2s = new ArrayList<>();
        
        int foldSize = data.size() / k;
        RegressionEvaluator evaluator = new RegressionEvaluator();
        
        for (int fold = 0; fold < k; fold++) {
            // Create train/test split for this fold
            int startIdx = fold * foldSize;
            int endIdx = (fold == k - 1) ? data.size() : (fold + 1) * foldSize;
            
            List<RegressionDataPoint> testFold = data.subList(startIdx, endIdx);
            List<RegressionDataPoint> trainFold = new ArrayList<>();
            trainFold.addAll(data.subList(0, startIdx));
            trainFold.addAll(data.subList(endIdx, data.size()));
            
            // Train and evaluate
            regressor.train(trainFold);
            
            List<Double> predictions = new ArrayList<>();
            List<Double> actuals = new ArrayList<>();
            
            for (RegressionDataPoint testPoint : testFold) {
                predictions.add(regressor.predict(testPoint.getFeatures()));
                actuals.add(testPoint.getTarget());
            }
            
            RegressionMetrics metrics = evaluator.evaluate(actuals, predictions);
            
            maes.add(metrics.getMAE());
            mses.add(metrics.getMSE());
            rmses.add(metrics.getRMSE());
            r2s.add(metrics.getR2());
        }
        
        return new RegressionCrossValidationResults(maes, mses, rmses, r2s);
    }
}
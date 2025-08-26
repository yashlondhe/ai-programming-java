package com.aiprogramming.ch03;

import java.util.*;

/**
 * Demonstrates data preprocessing specifically for customer churn prediction.
 */
public class ChurnDataPreprocessor {
    
    /**
     * Preprocesses the entire dataset for churn prediction
     */
    public Dataset preprocessData(Dataset dataset) {
        Dataset processed = dataset.copy();
        
        // Handle missing values
        processed = handleMissingValues(processed);
        
        // Encode categorical variables
        processed = encodeCategoricalVariables(processed);
        
        // Scale numerical features
        processed = scaleNumericalFeatures(processed);
        
        // Create new features
        processed = createDerivedFeatures(processed);
        
        return processed;
    }
    
    /**
     * Handles missing values in the dataset
     */
    private Dataset handleMissingValues(Dataset dataset) {
        // Fill missing total charges with monthly charges * tenure
        return dataset.map(point -> {
            if (point.getFeature("totalCharges") == null) {
                Object monthlyChargesObj = point.getFeature("monthlyCharges");
                Object tenureObj = point.getFeature("tenure");
                
                if (monthlyChargesObj instanceof Number && tenureObj instanceof Number) {
                    double monthlyCharges = ((Number) monthlyChargesObj).doubleValue();
                    int tenure = ((Number) tenureObj).intValue();
                    point.setFeature("totalCharges", monthlyCharges * tenure);
                }
            }
            return point;
        });
    }
    
    /**
     * Encodes categorical variables
     */
    private Dataset encodeCategoricalVariables(Dataset dataset) {
        // Encode gender
        dataset = encodeBinary(dataset, "gender", "Male", "Female");
        
        // Encode contract type
        dataset = encodeCategorical(dataset, "contractType", 
            Arrays.asList("Month-to-month", "One year", "Two year"));
        
        // Encode payment method
        dataset = encodeCategorical(dataset, "paymentMethod",
            Arrays.asList("Electronic check", "Mailed check", "Bank transfer", "Credit card"));
        
        // Encode location
        dataset = encodeCategorical(dataset, "location",
            Arrays.asList("Urban", "Suburban", "Rural"));
        
        return dataset;
    }
    
    /**
     * Encodes binary categorical variables
     */
    private Dataset encodeBinary(Dataset dataset, String featureName, String value1, String value2) {
        return dataset.map(point -> {
            Object value = point.getFeature(featureName);
            if (value instanceof String) {
                String strValue = (String) value;
                int encodedValue = strValue.equals(value1) ? 0 : 1;
                point.setFeature(featureName, encodedValue);
            }
            return point;
        });
    }
    
    /**
     * Encodes categorical variables with multiple values
     */
    private Dataset encodeCategorical(Dataset dataset, String featureName, List<String> categories) {
        return dataset.map(point -> {
            Object value = point.getFeature(featureName);
            if (value instanceof String) {
                String strValue = (String) value;
                int encodedValue = categories.indexOf(strValue);
                if (encodedValue == -1) {
                    encodedValue = categories.size(); // Default for unknown categories
                }
                point.setFeature(featureName, encodedValue);
            }
            return point;
        });
    }
    
    /**
     * Scales numerical features
     */
    private Dataset scaleNumericalFeatures(Dataset dataset) {
        String[] numericalFeatures = {"age", "tenure", "monthlyCharges", "totalCharges", "supportCalls"};
        
        FeatureScaler scaler = new FeatureScaler();
        for (String feature : numericalFeatures) {
            if (dataset.getFeatureNames().contains(feature)) {
                dataset = scaler.standardize(dataset, feature);
            }
        }
        
        return dataset;
    }
    
    /**
     * Creates derived features
     */
    private Dataset createDerivedFeatures(Dataset dataset) {
        return dataset.map(point -> {
            // Average monthly charges
            Object monthlyChargesObj = point.getFeature("monthlyCharges");
            Object tenureObj = point.getFeature("tenure");
            
            if (monthlyChargesObj instanceof Number && tenureObj instanceof Number) {
                double monthlyCharges = ((Number) monthlyChargesObj).doubleValue();
                int tenure = ((Number) tenureObj).intValue();
                double avgMonthlyCharges = tenure > 0 ? monthlyCharges / tenure : monthlyCharges;
                point.setFeature("avgMonthlyCharges", avgMonthlyCharges);
            }
            
            // Tenure to age ratio
            Object ageObj = point.getFeature("age");
            if (ageObj instanceof Number && tenureObj instanceof Number) {
                int age = ((Number) ageObj).intValue();
                int tenure = ((Number) tenureObj).intValue();
                double tenureToAgeRatio = age > 0 ? (double) tenure / age : 0.0;
                point.setFeature("tenureToAgeRatio", tenureToAgeRatio);
            }
            
            // High support calls flag
            Object supportCallsObj = point.getFeature("supportCalls");
            if (supportCallsObj instanceof Number) {
                int supportCalls = ((Number) supportCallsObj).intValue();
                point.setFeature("highSupportCalls", supportCalls > 3 ? 1 : 0);
            }
            
            // Contract length indicator
            Object contractTypeObj = point.getFeature("contractType");
            if (contractTypeObj instanceof Number) {
                int contractType = ((Number) contractTypeObj).intValue();
                point.setFeature("isLongTermContract", contractType > 0 ? 1 : 0);
            }
            
            // Payment method indicator
            Object paymentMethodObj = point.getFeature("paymentMethod");
            if (paymentMethodObj instanceof Number) {
                int paymentMethod = ((Number) paymentMethodObj).intValue();
                point.setFeature("isElectronicPayment", paymentMethod == 0 || paymentMethod == 2 ? 1 : 0);
            }
            
            return point;
        });
    }
    
    /**
     * Removes unnecessary features
     */
    public Dataset removeUnnecessaryFeatures(Dataset dataset) {
        List<String> featuresToRemove = Arrays.asList("customerId");
        
        return dataset.map(point -> {
            for (String feature : featuresToRemove) {
                point.getFeatures().remove(feature);
            }
            return point;
        });
    }
    
    /**
     * Handles outliers in numerical features
     */
    public Dataset handleOutliers(Dataset dataset) {
        OutlierDetector outlierDetector = new OutlierDetector();
        
        String[] numericalFeatures = {"age", "tenure", "monthlyCharges", "totalCharges", "supportCalls"};
        
        for (String feature : numericalFeatures) {
            if (dataset.getFeatureNames().contains(feature)) {
                // Cap outliers using IQR method
                dataset = outlierDetector.capOutliers(dataset, feature, 5.0, 95.0);
            }
        }
        
        return dataset;
    }
    
    /**
     * Creates feature interactions
     */
    public Dataset createFeatureInteractions(Dataset dataset) {
        FeatureCreator featureCreator = new FeatureCreator();
        
        // Create interaction between tenure and monthly charges
        if (dataset.getFeatureNames().contains("tenure") && dataset.getFeatureNames().contains("monthlyCharges")) {
            dataset = featureCreator.createInteractionFeatures(dataset, "tenure", "monthlyCharges");
        }
        
        // Create interaction between age and support calls
        if (dataset.getFeatureNames().contains("age") && dataset.getFeatureNames().contains("supportCalls")) {
            dataset = featureCreator.createInteractionFeatures(dataset, "age", "supportCalls");
        }
        
        return dataset;
    }
    
    /**
     * Performs feature selection
     */
    public Dataset performFeatureSelection(Dataset dataset, String targetFeature) {
        FeatureSelector selector = new FeatureSelector();
        
        // Select features by correlation
        List<String> selectedFeatures = selector.selectByCorrelation(dataset, targetFeature, 0.05);
        
        // Keep only selected features plus target
        return dataset.map(point -> {
            Map<String, Object> newFeatures = new HashMap<>();
            
            // Keep target feature
            if (point.hasTarget()) {
                newFeatures.put(targetFeature, point.getTarget());
            }
            
            // Keep selected features
            for (String feature : selectedFeatures) {
                if (point.hasFeature(feature)) {
                    newFeatures.put(feature, point.getFeature(feature));
                }
            }
            
            point.getFeatures().clear();
            point.getFeatures().putAll(newFeatures);
            
            return point;
        });
    }
    
    /**
     * Generates preprocessing report
     */
    public void generatePreprocessingReport(Dataset originalDataset, Dataset processedDataset) {
        System.out.println("=== Data Preprocessing Report ===");
        
        System.out.printf("Original dataset size: %d%n", originalDataset.size());
        System.out.printf("Processed dataset size: %d%n", processedDataset.size());
        
        System.out.printf("Original features: %s%n", originalDataset.getFeatureNames());
        System.out.printf("Processed features: %s%n", processedDataset.getFeatureNames());
        
        // Check for missing values
        long originalMissing = originalDataset.getDataPoints().stream()
                .filter(point -> point.getFeatures().values().stream().anyMatch(Objects::isNull))
                .count();
        long processedMissing = processedDataset.getDataPoints().stream()
                .filter(point -> point.getFeatures().values().stream().anyMatch(Objects::isNull))
                .count();
        
        System.out.printf("Missing values - Original: %d, Processed: %d%n", originalMissing, processedMissing);
        
        // Feature statistics
        System.out.println("\nFeature Statistics (Processed Dataset):");
        for (String feature : processedDataset.getNumericalFeatures()) {
            double mean = processedDataset.calculateMean(feature);
            double std = processedDataset.calculateStandardDeviation(feature);
            System.out.printf("%s: mean=%.4f, std=%.4f%n", feature, mean, std);
        }
    }
}

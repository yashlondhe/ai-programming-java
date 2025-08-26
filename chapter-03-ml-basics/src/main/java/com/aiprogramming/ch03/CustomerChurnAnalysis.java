package com.aiprogramming.ch03;

import java.io.*;
import java.util.*;

/**
 * Demonstrates a complete machine learning pipeline for customer churn prediction.
 * This class shows how to load, preprocess, and analyze customer data for churn prediction.
 */
public class CustomerChurnAnalysis {
    
    /**
     * Loads customer data from a CSV file
     */
    public Dataset loadCustomerData(String filePath) {
        List<DataPoint> dataPoints = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line = reader.readLine(); // Skip header
            while ((line = reader.readLine()) != null) {
                DataPoint point = parseCustomerData(line);
                if (point != null) {
                    dataPoints.add(point);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            // Return sample data if file not found
            return createSampleCustomerData();
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Parses a line of customer data from CSV
     */
    private DataPoint parseCustomerData(String line) {
        try {
            String[] fields = line.split(",");
            
            if (fields.length < 10) {
                return null; // Skip invalid lines
            }
            
            // Parse features
            int customerId = Integer.parseInt(fields[0].trim());
            String gender = fields[1].trim();
            int age = Integer.parseInt(fields[2].trim());
            String location = fields[3].trim();
            int tenure = Integer.parseInt(fields[4].trim());
            double monthlyCharges = Double.parseDouble(fields[5].trim());
            double totalCharges = Double.parseDouble(fields[6].trim());
            String contractType = fields[7].trim();
            String paymentMethod = fields[8].trim();
            int supportCalls = Integer.parseInt(fields[9].trim());
            boolean hasChurned = fields.length > 10 && fields[10].trim().equalsIgnoreCase("Yes");
            
            // Create feature vector
            Map<String, Object> features = new HashMap<>();
            features.put("customerId", customerId);
            features.put("gender", gender);
            features.put("age", age);
            features.put("location", location);
            features.put("tenure", tenure);
            features.put("monthlyCharges", monthlyCharges);
            features.put("totalCharges", totalCharges);
            features.put("contractType", contractType);
            features.put("paymentMethod", paymentMethod);
            features.put("supportCalls", supportCalls);
            
            return new DataPoint(features, hasChurned);
            
        } catch (NumberFormatException e) {
            System.err.println("Error parsing line: " + line);
            return null;
        }
    }
    
    /**
     * Creates sample customer data for demonstration
     */
    private Dataset createSampleCustomerData() {
        List<DataPoint> dataPoints = new ArrayList<>();
        Random random = new Random(42);
        
        String[] genders = {"Male", "Female"};
        String[] locations = {"Urban", "Suburban", "Rural"};
        String[] contractTypes = {"Month-to-month", "One year", "Two year"};
        String[] paymentMethods = {"Electronic check", "Mailed check", "Bank transfer", "Credit card"};
        
        for (int i = 1; i <= 1000; i++) {
            Map<String, Object> features = new HashMap<>();
            
            features.put("customerId", i);
            features.put("gender", genders[random.nextInt(genders.length)]);
            features.put("age", 18 + random.nextInt(62)); // 18-80 years
            features.put("location", locations[random.nextInt(locations.length)]);
            features.put("tenure", random.nextInt(72)); // 0-72 months
            features.put("monthlyCharges", 20.0 + random.nextDouble() * 100.0); // $20-$120
            features.put("totalCharges", 0.0); // Will be calculated
            features.put("contractType", contractTypes[random.nextInt(contractTypes.length)]);
            features.put("paymentMethod", paymentMethods[random.nextInt(paymentMethods.length)]);
            features.put("supportCalls", random.nextInt(10)); // 0-9 calls
            
            // Calculate total charges based on tenure and monthly charges
            double monthlyCharges = (Double) features.get("monthlyCharges");
            int tenure = (Integer) features.get("tenure");
            features.put("totalCharges", monthlyCharges * tenure);
            
            // Determine churn based on some business logic
            boolean hasChurned = determineChurn(features, random);
            
            dataPoints.add(new DataPoint(features, hasChurned));
        }
        
        return new Dataset(dataPoints);
    }
    
    /**
     * Determines if a customer has churned based on business rules
     */
    private boolean determineChurn(Map<String, Object> features, Random random) {
        double churnProbability = 0.0;
        
        // Higher churn probability for month-to-month contracts
        if ("Month-to-month".equals(features.get("contractType"))) {
            churnProbability += 0.3;
        }
        
        // Higher churn probability for high support calls
        int supportCalls = (Integer) features.get("supportCalls");
        if (supportCalls > 5) {
            churnProbability += 0.4;
        } else if (supportCalls > 2) {
            churnProbability += 0.2;
        }
        
        // Higher churn probability for high monthly charges
        double monthlyCharges = (Double) features.get("monthlyCharges");
        if (monthlyCharges > 80) {
            churnProbability += 0.2;
        }
        
        // Lower churn probability for longer tenure
        int tenure = (Integer) features.get("tenure");
        if (tenure > 24) {
            churnProbability -= 0.3;
        } else if (tenure > 12) {
            churnProbability -= 0.1;
        }
        
        // Add some randomness
        churnProbability += random.nextDouble() * 0.2 - 0.1;
        
        return random.nextDouble() < Math.max(0.0, Math.min(1.0, churnProbability));
    }
    
    /**
     * Explores and analyzes the customer data
     */
    public void exploreData(Dataset dataset) {
        System.out.println("=== Customer Churn Dataset Exploration ===");
        System.out.println("Total customers: " + dataset.size());
        System.out.println("Features: " + dataset.getFeatureNames());
        
        // Churn distribution
        long churnedCount = dataset.getDataPoints().stream()
                .filter(point -> (Boolean) point.getTarget())
                .count();
        double churnRate = (double) churnedCount / dataset.size();
        System.out.printf("Churn rate: %.2f%% (%d out of %d customers)%n", 
                churnRate * 100, churnedCount, dataset.size());
        
        // Feature statistics
        System.out.println("\nFeature Statistics:");
        for (String feature : dataset.getNumericalFeatures()) {
            double mean = dataset.calculateMean(feature);
            double std = dataset.calculateStandardDeviation(feature);
            double min = dataset.getMin(feature);
            double max = dataset.getMax(feature);
            System.out.printf("%s: mean=%.2f, std=%.2f, min=%.2f, max=%.2f%n", 
                    feature, mean, std, min, max);
        }
        
        // Categorical feature analysis
        System.out.println("\nCategorical Feature Analysis:");
        for (String feature : dataset.getCategoricalFeatures()) {
            Map<Object, Long> distribution = dataset.getDataPoints().stream()
                    .collect(java.util.stream.Collectors.groupingBy(
                            point -> point.getFeature(feature),
                            java.util.stream.Collectors.counting()
                    ));
            System.out.println(feature + ": " + distribution);
        }
        
        // Churn analysis by features
        System.out.println("\nChurn Analysis by Features:");
        analyzeChurnByFeature(dataset, "contractType");
        analyzeChurnByFeature(dataset, "paymentMethod");
        analyzeChurnByFeature(dataset, "location");
        analyzeChurnByFeature(dataset, "gender");
    }
    
    /**
     * Analyzes churn rate by a specific categorical feature
     */
    private void analyzeChurnByFeature(Dataset dataset, String featureName) {
        Map<Object, List<DataPoint>> grouped = dataset.getDataPoints().stream()
                .collect(java.util.stream.Collectors.groupingBy(
                        point -> point.getFeature(featureName)
                ));
        
        System.out.println("\n" + featureName + " churn analysis:");
        for (Map.Entry<Object, List<DataPoint>> entry : grouped.entrySet()) {
            Object value = entry.getKey();
            List<DataPoint> group = entry.getValue();
            
            long churnedInGroup = group.stream()
                    .filter(point -> (Boolean) point.getTarget())
                    .count();
            
            double churnRate = (double) churnedInGroup / group.size();
            System.out.printf("  %s: %.2f%% (%d/%d)%n", 
                    value, churnRate * 100, churnedInGroup, group.size());
        }
    }
    
    /**
     * Creates a sample customer data file for testing
     */
    public void createSampleDataFile(String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            // Write header
            writer.println("customerId,gender,age,location,tenure,monthlyCharges,totalCharges,contractType,paymentMethod,supportCalls,churned");
            
            // Write sample data
            Dataset sampleData = createSampleCustomerData();
            for (DataPoint point : sampleData.getDataPoints()) {
                Map<String, Object> features = point.getFeatures();
                boolean churned = (Boolean) point.getTarget();
                
                writer.printf("%d,%s,%d,%s,%d,%.2f,%.2f,%s,%s,%d,%s%n",
                        features.get("customerId"),
                        features.get("gender"),
                        features.get("age"),
                        features.get("location"),
                        features.get("tenure"),
                        features.get("monthlyCharges"),
                        features.get("totalCharges"),
                        features.get("contractType"),
                        features.get("paymentMethod"),
                        features.get("supportCalls"),
                        churned ? "Yes" : "No"
                );
            }
            
            System.out.println("Sample data file created: " + filePath);
            
        } catch (IOException e) {
            System.err.println("Error creating sample data file: " + e.getMessage());
        }
    }
    
    /**
     * Performs correlation analysis between features and churn
     */
    public void performCorrelationAnalysis(Dataset dataset) {
        System.out.println("\n=== Correlation Analysis ===");
        
        for (String feature : dataset.getNumericalFeatures()) {
            if (!"customerId".equals(feature)) {
                double correlation = calculateCorrelationWithChurn(dataset, feature);
                System.out.printf("Correlation between %s and churn: %.4f%n", feature, correlation);
            }
        }
    }
    
    /**
     * Calculates correlation between a numerical feature and churn
     */
    private double calculateCorrelationWithChurn(Dataset dataset, String featureName) {
        List<DataPoint> dataPoints = dataset.getDataPoints();
        
        // Calculate means
        double featureMean = dataset.calculateMean(featureName);
        double churnMean = dataPoints.stream()
                .mapToDouble(point -> (Boolean) point.getTarget() ? 1.0 : 0.0)
                .average()
                .orElse(0.0);
        
        // Calculate correlation
        double numerator = 0.0;
        double featureSumSquares = 0.0;
        double churnSumSquares = 0.0;
        
        for (DataPoint point : dataPoints) {
            double featureValue = point.getNumericalFeature(featureName);
            double churnValue = (Boolean) point.getTarget() ? 1.0 : 0.0;
            
            double featureDiff = featureValue - featureMean;
            double churnDiff = churnValue - churnMean;
            
            numerator += featureDiff * churnDiff;
            featureSumSquares += featureDiff * featureDiff;
            churnSumSquares += churnDiff * churnDiff;
        }
        
        if (featureSumSquares == 0 || churnSumSquares == 0) {
            return 0.0;
        }
        
        return numerator / Math.sqrt(featureSumSquares * churnSumSquares);
    }
}

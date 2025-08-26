package com.aiprogramming.ch04;

import java.util.*;

/**
 * Utility class for splitting classification datasets into training and test sets.
 * Maintains class distribution in both sets.
 */
public class ClassificationDataSplitter {
    
    private final double trainRatio;
    private final Random random;
    
    public ClassificationDataSplitter(double trainRatio) {
        this(trainRatio, new Random(42));
    }
    
    public ClassificationDataSplitter(double trainRatio, Random random) {
        if (trainRatio <= 0.0 || trainRatio >= 1.0) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        this.trainRatio = trainRatio;
        this.random = random;
    }
    
    /**
     * Splits the data into training and test sets while maintaining class distribution
     */
    public ClassificationDataSplit split(List<ClassificationDataPoint> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        
        // Group data by class
        Map<String, List<ClassificationDataPoint>> dataByClass = new HashMap<>();
        for (ClassificationDataPoint point : data) {
            dataByClass.computeIfAbsent(point.getLabel(), k -> new ArrayList<>()).add(point);
        }
        
        List<ClassificationDataPoint> trainingData = new ArrayList<>();
        List<ClassificationDataPoint> testData = new ArrayList<>();
        
        // Split each class proportionally
        for (List<ClassificationDataPoint> classData : dataByClass.values()) {
            Collections.shuffle(classData, random);
            
            int trainSize = (int) (classData.size() * trainRatio);
            
            trainingData.addAll(classData.subList(0, trainSize));
            testData.addAll(classData.subList(trainSize, classData.size()));
        }
        
        return new ClassificationDataSplit(trainingData, testData);
    }
    
    /**
     * Gets the training ratio
     */
    public double getTrainRatio() {
        return trainRatio;
    }
}

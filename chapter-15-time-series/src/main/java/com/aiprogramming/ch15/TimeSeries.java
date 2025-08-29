package com.aiprogramming.ch15;

import com.aiprogramming.utils.StatisticsUtils;
import com.aiprogramming.utils.ValidationUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Represents a time series with timestamps and corresponding values
 */
public class TimeSeries {
    private List<Long> timestamps;
    private List<Double> values;
    private String name;
    
    public TimeSeries(String name) {
        ValidationUtils.validateNotNull(name, "name");
        ValidationUtils.validateNonEmptyString(name, "name");
        this.name = name;
        this.timestamps = new ArrayList<>();
        this.values = new ArrayList<>();
    }
    
    public TimeSeries(String name, List<Long> timestamps, List<Double> values) {
        ValidationUtils.validateNotNull(name, "name");
        ValidationUtils.validateNonEmptyString(name, "name");
        ValidationUtils.validateNotNull(timestamps, "timestamps");
        ValidationUtils.validateNotNull(values, "values");
        
        if (timestamps.size() != values.size()) {
            throw new IllegalArgumentException("Timestamps and values must have the same size");
        }
        
        this.name = name;
        this.timestamps = new ArrayList<>(timestamps);
        this.values = new ArrayList<>(values);
    }
    
    /**
     * Add a data point to the time series
     */
    public void addPoint(long timestamp, double value) {
        ValidationUtils.validateFiniteValues(new double[]{value}, "value");
        timestamps.add(timestamp);
        values.add(value);
    }
    
    /**
     * Get the value at a specific index
     */
    public double getValue(int index) {
        if (index < 0 || index >= values.size()) {
            throw new IllegalArgumentException("Index must be between 0 and " + (values.size() - 1));
        }
        return values.get(index);
    }
    
    /**
     * Get the timestamp at a specific index
     */
    public long getTimestamp(int index) {
        if (index < 0 || index >= timestamps.size()) {
            throw new IllegalArgumentException("Index must be between 0 and " + (timestamps.size() - 1));
        }
        return timestamps.get(index);
    }
    
    /**
     * Get the size of the time series
     */
    public int size() {
        return values.size();
    }
    
    /**
     * Get all values as an array
     */
    public double[] getValues() {
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Get all timestamps as an array
     */
    public long[] getTimestamps() {
        return timestamps.stream().mapToLong(Long::longValue).toArray();
    }
    
    /**
     * Get a subset of the time series
     */
    public TimeSeries getSubset(int startIndex, int endIndex) {
        if (startIndex < 0 || startIndex >= values.size()) {
            throw new IllegalArgumentException("Start index must be between 0 and " + (values.size() - 1));
        }
        if (endIndex < 0 || endIndex >= values.size()) {
            throw new IllegalArgumentException("End index must be between 0 and " + (values.size() - 1));
        }
        if (startIndex >= endIndex) {
            throw new IllegalArgumentException("Start index must be less than end index");
        }
        
        List<Long> subTimestamps = timestamps.subList(startIndex, endIndex);
        List<Double> subValues = values.subList(startIndex, endIndex);
        return new TimeSeries(name + "_subset", subTimestamps, subValues);
    }
    
    /**
     * Calculate the mean of all values
     */
    public double getMean() {
        if (values.isEmpty()) {
            return 0.0;
        }
        double[] valuesArray = getValues();
        return StatisticsUtils.mean(valuesArray);
    }
    
    /**
     * Calculate the standard deviation of all values
     */
    public double getStandardDeviation() {
        if (values.isEmpty()) {
            return 0.0;
        }
        double[] valuesArray = getValues();
        return StatisticsUtils.standardDeviation(valuesArray);
    }
    
    /**
     * Get the minimum value
     */
    public double getMin() {
        if (values.isEmpty()) {
            return 0.0;
        }
        return values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    }
    
    /**
     * Get the maximum value
     */
    public double getMax() {
        if (values.isEmpty()) {
            return 0.0;
        }
        return values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
    }
    
    /**
     * Calculate the variance of all values
     */
    public double getVariance() {
        if (values.isEmpty()) {
            return 0.0;
        }
        double[] valuesArray = getValues();
        return StatisticsUtils.variance(valuesArray);
    }
    
    /**
     * Calculate the median of all values
     */
    public double getMedian() {
        if (values.isEmpty()) {
            return 0.0;
        }
        double[] valuesArray = getValues();
        return StatisticsUtils.median(valuesArray);
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    @Override
    public String toString() {
        return String.format("TimeSeries{name='%s', size=%d, mean=%.2f, std=%.2f}", 
                           name, size(), getMean(), getStandardDeviation());
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TimeSeries that = (TimeSeries) o;
        return Objects.equals(timestamps, that.timestamps) &&
               Objects.equals(values, that.values) &&
               Objects.equals(name, that.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(timestamps, values, name);
    }
}

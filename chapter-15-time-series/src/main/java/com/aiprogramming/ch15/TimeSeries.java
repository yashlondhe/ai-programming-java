package com.aiprogramming.ch15;

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
        this.name = name;
        this.timestamps = new ArrayList<>();
        this.values = new ArrayList<>();
    }
    
    public TimeSeries(String name, List<Long> timestamps, List<Double> values) {
        this.name = name;
        this.timestamps = new ArrayList<>(timestamps);
        this.values = new ArrayList<>(values);
    }
    
    /**
     * Add a data point to the time series
     */
    public void addPoint(long timestamp, double value) {
        timestamps.add(timestamp);
        values.add(value);
    }
    
    /**
     * Get the value at a specific index
     */
    public double getValue(int index) {
        return values.get(index);
    }
    
    /**
     * Get the timestamp at a specific index
     */
    public long getTimestamp(int index) {
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
        List<Long> subTimestamps = timestamps.subList(startIndex, endIndex);
        List<Double> subValues = values.subList(startIndex, endIndex);
        return new TimeSeries(name + "_subset", subTimestamps, subValues);
    }
    
    /**
     * Calculate the mean of all values
     */
    public double getMean() {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculate the standard deviation of all values
     */
    public double getStandardDeviation() {
        double mean = getMean();
        double variance = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    /**
     * Get the minimum value
     */
    public double getMin() {
        return values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    }
    
    /**
     * Get the maximum value
     */
    public double getMax() {
        return values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
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

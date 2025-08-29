package com.aiprogramming.ch16.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;
import java.time.ZoneId;

/**
 * Statistics and performance metrics for an AI model
 */
public class ModelStats {

    @JsonProperty("total_predictions")
    private long totalPredictions;

    @JsonProperty("average_prediction_time_ms")
    private double averagePredictionTimeMs;

    @JsonProperty("last_prediction_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime lastPredictionTime;

    @JsonProperty("accuracy")
    private Double accuracy;

    @JsonProperty("throughput_predictions_per_second")
    private double throughputPredictionsPerSecond;

    @JsonProperty("memory_usage_mb")
    private double memoryUsageMb;

    @JsonProperty("cpu_usage_percent")
    private double cpuUsagePercent;

    @JsonProperty("error_rate")
    private double errorRate;

    public ModelStats() {
    }

    public ModelStats(long totalPredictions, double averagePredictionTimeMs, 
                     long lastPredictionTime, Double accuracy) {
        this.totalPredictions = totalPredictions;
        this.averagePredictionTimeMs = averagePredictionTimeMs;
        this.lastPredictionTime = LocalDateTime.ofInstant(
            java.time.Instant.ofEpochMilli(lastPredictionTime), 
            ZoneId.systemDefault()
        );
        this.accuracy = accuracy;
        
        // Calculate derived metrics
        if (averagePredictionTimeMs > 0) {
            this.throughputPredictionsPerSecond = 1000.0 / averagePredictionTimeMs;
        }
    }

    // Getters and Setters
    public long getTotalPredictions() {
        return totalPredictions;
    }

    public void setTotalPredictions(long totalPredictions) {
        this.totalPredictions = totalPredictions;
    }

    public double getAveragePredictionTimeMs() {
        return averagePredictionTimeMs;
    }

    public void setAveragePredictionTimeMs(double averagePredictionTimeMs) {
        this.averagePredictionTimeMs = averagePredictionTimeMs;
    }

    public LocalDateTime getLastPredictionTime() {
        return lastPredictionTime;
    }

    public void setLastPredictionTime(LocalDateTime lastPredictionTime) {
        this.lastPredictionTime = lastPredictionTime;
    }

    public Double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }

    public double getThroughputPredictionsPerSecond() {
        return throughputPredictionsPerSecond;
    }

    public void setThroughputPredictionsPerSecond(double throughputPredictionsPerSecond) {
        this.throughputPredictionsPerSecond = throughputPredictionsPerSecond;
    }

    public double getMemoryUsageMb() {
        return memoryUsageMb;
    }

    public void setMemoryUsageMb(double memoryUsageMb) {
        this.memoryUsageMb = memoryUsageMb;
    }

    public double getCpuUsagePercent() {
        return cpuUsagePercent;
    }

    public void setCpuUsagePercent(double cpuUsagePercent) {
        this.cpuUsagePercent = cpuUsagePercent;
    }

    public double getErrorRate() {
        return errorRate;
    }

    public void setErrorRate(double errorRate) {
        this.errorRate = errorRate;
    }

    @Override
    public String toString() {
        return String.format(
            "ModelStats{totalPredictions=%d, avgTime=%.2fms, throughput=%.2f/sec, accuracy=%.4f}",
            totalPredictions, averagePredictionTimeMs, throughputPredictionsPerSecond, accuracy
        );
    }
}

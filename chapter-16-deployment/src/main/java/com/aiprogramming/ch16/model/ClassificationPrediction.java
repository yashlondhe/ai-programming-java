package com.aiprogramming.ch16.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Arrays;

/**
 * Represents the result of a classification prediction
 */
public class ClassificationPrediction {

    @JsonProperty("predicted_class")
    private String predictedClass;

    @JsonProperty("confidence")
    private double confidence;

    @JsonProperty("features")
    private double[] features;

    @JsonProperty("prediction_id")
    private String predictionId;

    @JsonProperty("model_id")
    private String modelId;

    @JsonProperty("timestamp")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime timestamp;

    @JsonProperty("processing_time_ms")
    private long processingTimeMs;

    public ClassificationPrediction() {
        this.timestamp = LocalDateTime.now();
    }

    public ClassificationPrediction(String predictedClass, double confidence, double[] features, long timestamp) {
        this();
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.features = features != null ? Arrays.copyOf(features, features.length) : null;
        this.timestamp = LocalDateTime.ofInstant(
            java.time.Instant.ofEpochMilli(timestamp), 
            ZoneId.systemDefault()
        );
    }

    // Getters and Setters
    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public double[] getFeatures() {
        return features != null ? Arrays.copyOf(features, features.length) : null;
    }

    public void setFeatures(double[] features) {
        this.features = features != null ? Arrays.copyOf(features, features.length) : null;
    }

    public String getPredictionId() {
        return predictionId;
    }

    public void setPredictionId(String predictionId) {
        this.predictionId = predictionId;
    }

    public String getModelId() {
        return modelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }

    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }

    @Override
    public String toString() {
        return String.format(
            "ClassificationPrediction{predictedClass='%s', confidence=%.4f, timestamp=%s}",
            predictedClass, confidence, timestamp
        );
    }
}

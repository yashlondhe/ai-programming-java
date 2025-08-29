package com.aiprogramming.ch16.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import javax.persistence.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Base class representing an AI/ML model with versioning and metadata
 */
@Entity
@Table(name = "ai_models")
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
@DiscriminatorColumn(name = "model_type")
public abstract class AIModel {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String modelId;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String version;

    @Column(nullable = false)
    private String description;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ModelStatus status = ModelStatus.DRAFT;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column(nullable = false)
    private LocalDateTime updatedAt;

    @Column(nullable = false)
    private String createdBy;

    @ElementCollection
    @CollectionTable(name = "model_metadata", joinColumns = @JoinColumn(name = "model_id"))
    @MapKeyColumn(name = "metadata_key")
    @Column(name = "metadata_value")
    private Map<String, String> metadata = new HashMap<>();

    @Column(columnDefinition = "TEXT")
    private String modelPath;

    @Column
    private Double accuracy;

    @Column
    private Long trainingTimeMs;

    @Column
    private Long modelSizeBytes;

    // Constructors
    public AIModel() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public AIModel(String modelId, String name, String version, String description) {
        this();
        this.modelId = modelId;
        this.name = name;
        this.version = version;
        this.description = description;
    }

    // Abstract methods that subclasses must implement
    @JsonIgnore
    public abstract String getModelType();

    @JsonIgnore
    public abstract boolean isLoaded();

    @JsonIgnore
    public abstract void loadModel();

    @JsonIgnore
    public abstract void unloadModel();

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getModelId() {
        return modelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public ModelStatus getStatus() {
        return status;
    }

    public void setStatus(ModelStatus status) {
        this.status = status;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    public String getCreatedBy() {
        return createdBy;
    }

    public void setCreatedBy(String createdBy) {
        this.createdBy = createdBy;
    }

    public Map<String, String> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, String> metadata) {
        this.metadata = metadata;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public Double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }

    public Long getTrainingTimeMs() {
        return trainingTimeMs;
    }

    public void setTrainingTimeMs(Long trainingTimeMs) {
        this.trainingTimeMs = trainingTimeMs;
    }

    public Long getModelSizeBytes() {
        return modelSizeBytes;
    }

    public void setModelSizeBytes(Long modelSizeBytes) {
        this.modelSizeBytes = modelSizeBytes;
    }

    // Utility methods
    public void addMetadata(String key, String value) {
        this.metadata.put(key, value);
    }

    public String getMetadata(String key) {
        return this.metadata.get(key);
    }

    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.SHORT_PREFIX_STYLE);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        AIModel aiModel = (AIModel) obj;
        return modelId != null && modelId.equals(aiModel.modelId);
    }

    @Override
    public int hashCode() {
        return modelId != null ? modelId.hashCode() : 0;
    }
}

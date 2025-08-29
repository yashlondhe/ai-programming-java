# Chapter 16: Deploying AI Applications

## Introduction

Deploying AI/ML models in production is a critical step in the machine learning lifecycle. Unlike traditional software deployment, AI model deployment involves unique challenges such as model versioning, performance monitoring, A/B testing, and maintaining model accuracy over time. This chapter covers the essential concepts and practical implementation of deploying AI applications using Java and Spring Boot.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand AI model deployment concepts and best practices
- Implement REST APIs for AI model inference using Spring Boot
- Handle model versioning, lifecycle management, and A/B testing
- Containerize AI applications using Docker
- Monitor and maintain AI systems in production
- Implement model serving architectures and scalability considerations

### Key Concepts

- **Model Deployment**: The process of making trained models available for predictions in production
- **Model Serving**: The infrastructure and APIs that handle model inference requests
- **Model Versioning**: Managing different versions of models and their lifecycle
- **A/B Testing**: Comparing different model versions in production
- **Model Monitoring**: Tracking model performance, accuracy, and system health
- **Containerization**: Packaging applications with dependencies for consistent deployment
- **Microservices**: Architectural pattern for building scalable, maintainable applications

## 16.1 Model Deployment Fundamentals

Model deployment is the process of taking a trained machine learning model and making it available for predictions in a production environment. This involves several key considerations:

### 16.1.1 Deployment Challenges

AI model deployment presents unique challenges compared to traditional software deployment:

1. **Model Size and Performance**: Large models may require significant computational resources
2. **Versioning Complexity**: Models need versioning for rollbacks and A/B testing
3. **Data Drift**: Model performance can degrade as data distributions change
4. **Scalability**: Prediction services must handle varying load
5. **Monitoring**: Need to track both system metrics and model performance

### 16.1.2 Deployment Architecture

A typical AI model deployment architecture consists of several components:

```java
package com.aiprogramming.ch16.model;

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

    // Abstract methods that subclasses must implement
    @JsonIgnore
    public abstract String getModelType();

    @JsonIgnore
    public abstract boolean isLoaded();

    @JsonIgnore
    public abstract void loadModel();

    @JsonIgnore
    public abstract void unloadModel();
}
```

## 16.2 Model Lifecycle Management

Managing the lifecycle of AI models is crucial for maintaining reliable prediction services. Models go through various states from development to retirement.

### 16.2.1 Model Status Enum

```java
package com.aiprogramming.ch16.model;

/**
 * Enum representing the lifecycle status of an AI model
 */
public enum ModelStatus {
    
    /**
     * Model is in development/testing phase
     */
    DRAFT("Draft"),
    
    /**
     * Model is being trained
     */
    TRAINING("Training"),
    
    /**
     * Model training completed, ready for evaluation
     */
    TRAINED("Trained"),
    
    /**
     * Model is being evaluated
     */
    EVALUATING("Evaluating"),
    
    /**
     * Model passed evaluation, ready for deployment
     */
    APPROVED("Approved"),
    
    /**
     * Model is deployed and serving predictions
     */
    DEPLOYED("Deployed"),
    
    /**
     * Model is in A/B testing
     */
    AB_TESTING("A/B Testing"),
    
    /**
     * Model is deprecated and will be removed
     */
    DEPRECATED("Deprecated"),
    
    /**
     * Model is archived and no longer in use
     */
    ARCHIVED("Archived"),
    
    /**
     * Model failed evaluation or deployment
     */
    FAILED("Failed");

    private final String displayName;

    ModelStatus(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Check if the model is in an active state (can serve predictions)
     */
    public boolean isActive() {
        return this == DEPLOYED || this == AB_TESTING;
    }

    /**
     * Check if the model can be deployed
     */
    public boolean canDeploy() {
        return this == APPROVED || this == DEPRECATED;
    }

    /**
     * Check if the model is in a terminal state
     */
    public boolean isTerminal() {
        return this == ARCHIVED || this == FAILED;
    }
}
```

### 16.2.2 Model Service Implementation

The ModelService handles the business logic for model lifecycle management:

```java
package com.aiprogramming.ch16.service;

/**
 * Service class for AI model management operations
 */
@Service
@Transactional
public class ModelService {

    private static final Logger logger = LoggerFactory.getLogger(ModelService.class);

    @Autowired
    private ModelRepository modelRepository;

    /**
     * Deploy a model (load it into memory and make it available for predictions)
     */
    public boolean deployModel(String modelId) {
        logger.info("Deploying model: {}", modelId);
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            logger.error("Model not found for deployment: {}", modelId);
            return false;
        }
        
        AIModel model = modelOpt.get();
        
        // Check if model can be deployed
        if (!model.getStatus().canDeploy()) {
            logger.error("Model {} cannot be deployed in current status: {}", 
                modelId, model.getStatus());
            return false;
        }
        
        try {
            // Load the model into memory
            model.loadModel();
            
            // Update status to deployed
            model.setStatus(ModelStatus.DEPLOYED);
            model.setUpdatedAt(LocalDateTime.now());
            modelRepository.save(model);
            
            logger.info("Successfully deployed model: {}", modelId);
            return true;
            
        } catch (Exception e) {
            logger.error("Failed to deploy model: {}", modelId, e);
            return false;
        }
    }

    /**
     * Update model status
     */
    public AIModel updateModelStatus(String modelId, ModelStatus newStatus) {
        logger.info("Updating model {} status to: {}", modelId, newStatus);
        
        Optional<AIModel> modelOpt = modelRepository.findByModelId(modelId);
        if (modelOpt.isEmpty()) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }
        
        AIModel model = modelOpt.get();
        model.setStatus(newStatus);
        model.setUpdatedAt(LocalDateTime.now());
        
        // Handle status-specific logic
        if (newStatus == ModelStatus.DEPLOYED) {
            handleModelDeployment(model);
        } else if (newStatus == ModelStatus.ARCHIVED) {
            handleModelArchival(model);
        }
        
        AIModel updatedModel = modelRepository.save(model);
        logger.info("Successfully updated model {} status to: {}", modelId, newStatus);
        
        return updatedModel;
    }
}
```

## 16.3 REST API Implementation

REST APIs are the primary interface for model serving in production environments. They provide a standardized way for clients to interact with deployed models.

### 16.3.1 Model Controller

The ModelController handles model management operations:

```java
package com.aiprogramming.ch16.controller;

/**
 * REST API controller for AI model management operations
 */
@RestController
@RequestMapping("/api/v1/models")
@CrossOrigin(origins = "*")
public class ModelController {

    private static final Logger logger = LoggerFactory.getLogger(ModelController.class);

    @Autowired
    private ModelService modelService;

    /**
     * Create a new AI model
     * 
     * POST /api/v1/models
     */
    @PostMapping
    public ResponseEntity<AIModel> createModel(@Valid @RequestBody AIModel model) {
        logger.info("Creating new model: {}", model.getName());
        
        try {
            AIModel createdModel = modelService.createModel(model);
            return ResponseEntity.status(HttpStatus.CREATED).body(createdModel);
        } catch (Exception e) {
            logger.error("Failed to create model", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Deploy a model
     * 
     * POST /api/v1/models/{modelId}/deploy
     */
    @PostMapping("/{modelId}/deploy")
    public ResponseEntity<String> deployModel(@PathVariable String modelId) {
        logger.info("Deploying model: {}", modelId);
        
        try {
            boolean success = modelService.deployModel(modelId);
            if (success) {
                return ResponseEntity.ok("Model deployed successfully");
            } else {
                return ResponseEntity.badRequest().body("Failed to deploy model");
            }
        } catch (Exception e) {
            logger.error("Failed to deploy model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get model statistics
     * 
     * GET /api/v1/models/{modelId}/stats
     */
    @GetMapping("/{modelId}/stats")
    public ResponseEntity<ModelStats> getModelStats(@PathVariable String modelId) {
        logger.info("Getting stats for model: {}", modelId);
        
        try {
            ModelStats stats = modelService.getModelStats(modelId);
            return ResponseEntity.ok(stats);
        } catch (IllegalArgumentException e) {
            logger.error("Model not found: {}", modelId);
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            logger.error("Failed to get model stats: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
```

### 16.3.2 Prediction Controller

The PredictionController handles model inference requests:

```java
package com.aiprogramming.ch16.controller;

/**
 * REST API controller for model prediction and inference operations
 */
@RestController
@RequestMapping("/api/v1/predictions")
@CrossOrigin(origins = "*")
public class PredictionController {

    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);

    @Autowired
    private PredictionService predictionService;

    /**
     * Make a prediction using a specific model
     * 
     * POST /api/v1/predictions/{modelId}
     */
    @PostMapping("/{modelId}")
    public ResponseEntity<ClassificationPrediction> predict(
            @PathVariable String modelId,
            @Valid @RequestBody PredictionRequest request) {
        
        logger.info("Making prediction with model: {}", modelId);
        
        try {
            // Validate input
            if (request.getFeatures() == null) {
                return ResponseEntity.badRequest().build();
            }

            // Make prediction
            ClassificationPrediction prediction = predictionService.predict(modelId, request.getFeatures());
            return ResponseEntity.ok(prediction);
            
        } catch (IllegalArgumentException e) {
            logger.error("Invalid prediction request: {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (IllegalStateException e) {
            logger.error("Model not ready for prediction: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).build();
        } catch (Exception e) {
            logger.error("Prediction failed for model: {}", modelId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Make a batch prediction using multiple models (A/B testing)
     * 
     * POST /api/v1/predictions/batch
     */
    @PostMapping("/batch")
    public ResponseEntity<List<ClassificationPrediction>> batchPredict(
            @Valid @RequestBody BatchPredictionRequest request) {
        
        logger.info("Making batch prediction with {} models", 
            request.getModelIds() != null ? request.getModelIds().size() : 0);
        
        try {
            // Validate input
            if (request.getModelIds() == null || request.getModelIds().isEmpty()) {
                return ResponseEntity.badRequest().build();
            }
            if (request.getFeatures() == null) {
                return ResponseEntity.badRequest().build();
            }

            // Make batch prediction
            List<ClassificationPrediction> predictions = predictionService.batchPredict(
                request.getModelIds(), request.getFeatures());
            return ResponseEntity.ok(predictions);
            
        } catch (Exception e) {
            logger.error("Batch prediction failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
```

## 16.4 Model Implementation

Concrete model implementations provide the actual prediction logic. Here we implement a classification model as an example.

### 16.4.1 Classification Model

```java
package com.aiprogramming.ch16.model;

/**
 * Concrete implementation of AIModel for classification tasks
 */
@Entity
@DiscriminatorValue("CLASSIFICATION")
public class ClassificationModel extends AIModel {

    private static final Logger logger = LoggerFactory.getLogger(ClassificationModel.class);
    
    // In-memory model storage (in production, this would be external)
    private static final Map<String, ClassificationModel> loadedModels = new ConcurrentHashMap<>();
    
    // Model parameters (simplified for demonstration)
    private double[] weights;
    private double bias;
    private List<String> featureNames;
    private List<String> classLabels;
    
    // Model state
    private boolean modelLoaded = false;
    private long lastPredictionTime = 0;
    private long totalPredictions = 0;
    private double averagePredictionTime = 0.0;

    @Override
    @JsonIgnore
    public String getModelType() {
        return "CLASSIFICATION";
    }

    @Override
    @JsonIgnore
    public boolean isLoaded() {
        return modelLoaded && loadedModels.containsKey(getModelId());
    }

    @Override
    @JsonIgnore
    public void loadModel() {
        if (isLoaded()) {
            logger.info("Model {} is already loaded", getModelId());
            return;
        }

        try {
            logger.info("Loading classification model: {}", getModelId());
            
            // Simulate model loading (in real implementation, load from file/database)
            initializeModel();
            
            // Register the loaded model
            loadedModels.put(getModelId(), this);
            modelLoaded = true;
            
            logger.info("Successfully loaded model: {}", getModelId());
            
        } catch (Exception e) {
            logger.error("Failed to load model: {}", getModelId(), e);
            throw new RuntimeException("Model loading failed", e);
        }
    }

    /**
     * Make a prediction using the loaded model
     */
    public ClassificationPrediction predict(double[] features) {
        if (!isLoaded()) {
            throw new IllegalStateException("Model is not loaded: " + getModelId());
        }

        long startTime = System.currentTimeMillis();
        
        try {
            // Validate input
            if (features == null || features.length != weights.length) {
                throw new IllegalArgumentException(
                    String.format("Expected %d features, got %d", 
                        weights.length, features == null ? 0 : features.length));
            }

            // Simple linear classification (for demonstration)
            double score = bias;
            for (int i = 0; i < features.length; i++) {
                score += weights[i] * features[i];
            }

            // Convert to probability using sigmoid
            double probability = 1.0 / (1.0 + Math.exp(-score));
            
            // Determine predicted class
            String predictedClass = probability > 0.5 ? classLabels.get(1) : classLabels.get(0);
            
            // Create prediction result
            ClassificationPrediction prediction = new ClassificationPrediction(
                predictedClass,
                probability,
                features,
                System.currentTimeMillis()
            );

            // Update statistics
            updatePredictionStats(System.currentTimeMillis() - startTime);
            
            return prediction;
            
        } catch (Exception e) {
            logger.error("Prediction failed for model: {}", getModelId(), e);
            throw new RuntimeException("Prediction failed", e);
        }
    }

    /**
     * Initialize model with sample data (for demonstration)
     */
    private void initializeModel() {
        // Simulate loading model parameters
        this.weights = new double[]{0.5, -0.3, 0.8, -0.2, 0.1};
        this.bias = -0.1;
        this.featureNames = List.of("feature1", "feature2", "feature3", "feature4", "feature5");
        this.classLabels = List.of("negative", "positive");
        
        // Update metadata
        addMetadata("algorithm", "logistic_regression");
        addMetadata("feature_count", String.valueOf(weights.length));
        addMetadata("class_count", String.valueOf(classLabels.size()));
    }
}
```

## 16.5 Containerization with Docker

Containerization is essential for consistent deployment across different environments. Docker provides a standardized way to package applications with their dependencies.

### 16.5.1 Dockerfile

```dockerfile
# Multi-stage build for AI Model Deployment Service
FROM maven:3.8.4-openjdk-11 AS build

# Set working directory
WORKDIR /app

# Copy pom.xml and download dependencies
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src

# Build the application
RUN mvn clean package -DskipTests

# Runtime stage
FROM openjdk:11-jre-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy the built JAR from build stage
COPY --from=build /app/target/chapter-16-deployment-1.0.0.jar app.jar

# Create directories for model storage
RUN mkdir -p /app/models /app/logs /tmp/ai-models && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ai-deployment/actuator/health || exit 1

# JVM options for production
ENV JAVA_OPTS="-Xms512m -Xmx2g -XX:+UseG1GC -XX:+UseContainerSupport"

# Run the application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
```

### 16.5.2 Docker Compose

Docker Compose simplifies multi-service deployment:

```yaml
version: '3.8'

services:
  ai-deployment-service:
    build: .
    container_name: ai-deployment-service
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - JAVA_OPTS=-Xms512m -Xmx2g -XX:+UseG1GC
    volumes:
      - ai-models:/app/models
      - ai-logs:/app/logs
    networks:
      - ai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/ai-deployment/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    container_name: ai-deployment-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - ai-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: ai-deployment-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - ai-network
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  ai-models:
    driver: local
  ai-logs:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  ai-network:
    driver: bridge
```

## 16.6 Monitoring and Observability

Monitoring is crucial for maintaining reliable AI systems in production. The application includes comprehensive monitoring capabilities.

### 16.6.1 Model Statistics

```java
package com.aiprogramming.ch16.model;

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
}
```

### 16.6.2 Spring Boot Actuator

The application uses Spring Boot Actuator for health checks and metrics:

```properties
# Actuator Configuration
management.endpoints.web.exposure.include=health,info,metrics,prometheus
management.endpoint.health.show-details=always
management.metrics.export.prometheus.enabled=true
```

## 16.7 A/B Testing

A/B testing allows comparing different model versions in production to determine which performs better.

### 16.7.1 Batch Prediction Service

```java
package com.aiprogramming.ch16.service;

/**
 * Service class for handling model predictions and inference
 */
@Service
public class PredictionService {

    @Autowired
    private ModelService modelService;

    /**
     * Make a batch prediction using multiple models (A/B testing)
     */
    public List<ClassificationPrediction> batchPredict(
            List<String> modelIds, double[] features) {
        
        logger.info("Making batch prediction with {} models", modelIds.size());
        
        return modelIds.stream()
            .map(modelId -> {
                try {
                    return predict(modelId, features);
                } catch (Exception e) {
                    logger.error("Failed to get prediction from model: {}", modelId, e);
                    return null;
                }
            })
            .filter(prediction -> prediction != null)
            .collect(java.util.stream.Collectors.toList());
    }

    /**
     * Make a prediction using a deployed model
     */
    public ClassificationPrediction predict(String modelId, double[] features) {
        logger.info("Making prediction with model: {}", modelId);
        
        // Get the model
        Optional<AIModel> modelOpt = modelService.getModelById(modelId);
        if (modelOpt.isEmpty()) {
            throw new IllegalArgumentException("Model not found: " + modelId);
        }
        
        AIModel model = modelOpt.get();
        
        // Check if model is active (deployed or A/B testing)
        if (!model.getStatus().isActive()) {
            throw new IllegalStateException(
                "Model is not active. Current status: " + model.getStatus());
        }
        
        // Check if model is loaded
        if (!model.isLoaded()) {
            logger.warn("Model {} is not loaded, attempting to load it", modelId);
            try {
                model.loadModel();
            } catch (Exception e) {
                throw new RuntimeException("Failed to load model for prediction", e);
            }
        }
        
        // Make prediction based on model type
        if (model instanceof ClassificationModel) {
            return predictWithClassificationModel((ClassificationModel) model, features);
        } else {
            throw new UnsupportedOperationException(
                "Prediction not supported for model type: " + model.getModelType());
        }
    }
}
```

## 16.8 Performance Optimization

Optimizing AI model serving performance is crucial for production systems.

### 16.8.1 JVM Tuning

```bash
# Production JVM options
export JAVA_OPTS="-Xms2g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### 16.8.2 Model Optimization Strategies

1. **Model Quantization**: Reduce model size and improve inference speed
2. **Batch Processing**: Process multiple predictions together
3. **Caching**: Cache frequently used predictions
4. **Async Processing**: Use async endpoints for long-running predictions

## 16.9 Security Considerations

Security is paramount in production AI systems.

### 16.9.1 Input Validation

```java
/**
 * Validate prediction input
 */
public void validatePredictionInput(double[] features, int expectedFeatureCount) {
    if (features == null) {
        throw new IllegalArgumentException("Features cannot be null");
    }
    
    if (features.length != expectedFeatureCount) {
        throw new IllegalArgumentException(
            String.format("Expected %d features, got %d", 
                expectedFeatureCount, features.length));
    }
    
    // Check for NaN or infinite values
    for (int i = 0; i < features.length; i++) {
        if (Double.isNaN(features[i]) || Double.isInfinite(features[i])) {
            throw new IllegalArgumentException(
                String.format("Invalid feature value at index %d: %f", i, features[i]));
        }
    }
}
```

### 16.9.2 Production Security Configuration

```properties
# Security Configuration (for production)
spring.security.user.name=admin
spring.security.user.password=secure_password
spring.security.basic.enabled=true

# HTTPS/TLS
server.ssl.enabled=true
server.ssl.key-store=classpath:keystore.p12
server.ssl.key-store-password=password
```

## 16.10 Deployment Strategies

Different deployment strategies offer various trade-offs between risk and complexity.

### 16.10.1 Blue-Green Deployment

```bash
# Deploy new version (green)
docker-compose -f docker-compose-green.yml up -d

# Switch traffic
curl -X PUT http://load-balancer/switch-to-green

# Decommission old version (blue)
docker-compose -f docker-compose-blue.yml down
```

### 16.10.2 Canary Deployment

```bash
# Deploy canary (10% traffic)
docker-compose -f docker-compose-canary.yml up -d

# Monitor metrics
curl http://localhost:9090/api/v1/query?query=prediction_latency

# Gradually increase traffic
curl -X PUT http://load-balancer/set-canary-weight/50
```

### 16.10.3 Rolling Updates

```bash
# Update with zero downtime
docker-compose up -d --no-deps --build ai-deployment-service
```

## 16.11 Testing and Validation

Comprehensive testing ensures reliable model deployment.

### 16.11.1 Unit Testing

```java
@SpringBootTest
class ModelServiceTest {

    @Autowired
    private ModelService modelService;

    @Test
    void testCreateModel() {
        // Test model creation
        ClassificationModel model = new ClassificationModel(
            "test-model", "Test Model", "1.0.0", "Test description");
        
        AIModel createdModel = modelService.createModel(model);
        
        assertNotNull(createdModel);
        assertEquals("test-model", createdModel.getModelId());
        assertEquals(ModelStatus.DRAFT, createdModel.getStatus());
    }

    @Test
    void testDeployModel() {
        // Test model deployment
        boolean success = modelService.deployModel("test-model");
        assertTrue(success);
    }
}
```

### 16.11.2 Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p test-data.json \
   http://localhost:8080/ai-deployment/api/v1/predictions/model_123
```

## 16.12 Troubleshooting and Maintenance

Effective troubleshooting and maintenance procedures are essential for production systems.

### 16.12.1 Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model files
   ls -la /tmp/ai-models/
   
   # Check logs
   docker logs ai-deployment-service
   ```

2. **High Memory Usage**
   ```bash
   # Check JVM memory
   jstat -gc <pid>
   
   # Check container memory
   docker stats ai-deployment-service
   ```

3. **Slow Predictions**
   ```bash
   # Check prediction latency
   curl http://localhost:8080/ai-deployment/actuator/metrics/prediction.latency
   
   # Check CPU usage
   top -p <pid>
   ```

### 16.12.2 Monitoring Dashboards

The application includes Grafana dashboards for monitoring:

- Model performance metrics
- System resource utilization
- Prediction latency and throughput
- Error rates and availability

## Summary

This chapter covered the essential aspects of deploying AI applications in production:

1. **Model Deployment Fundamentals**: Understanding the challenges and architecture of AI model deployment
2. **Model Lifecycle Management**: Managing model versions and status transitions
3. **REST API Implementation**: Building scalable APIs for model serving
4. **Containerization**: Using Docker for consistent deployment
5. **Monitoring and Observability**: Tracking model and system performance
6. **A/B Testing**: Comparing model versions in production
7. **Performance Optimization**: Tuning for production workloads
8. **Security**: Implementing security best practices
9. **Deployment Strategies**: Blue-green, canary, and rolling deployments
10. **Testing and Maintenance**: Ensuring reliability and troubleshooting

The implementation demonstrates a complete, production-ready AI model deployment system using Java and Spring Boot, with comprehensive monitoring, containerization, and deployment strategies.

## Exercises

1. **Model Deployment Pipeline**: Implement a complete CI/CD pipeline for model deployment
2. **A/B Testing Framework**: Build a more sophisticated A/B testing system with traffic splitting
3. **Model Performance Monitoring**: Create custom metrics and alerts for model drift detection
4. **Multi-Model Serving**: Extend the system to support different model types (regression, clustering)
5. **Load Balancing**: Implement load balancing for multiple model instances
6. **Model Rollback**: Add automatic rollback capabilities for failed deployments
7. **Data Pipeline Integration**: Connect the deployment system with data pipelines
8. **Cost Optimization**: Implement cost-aware model serving and resource allocation

## Further Reading

- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [MLOps: Continuous Delivery for Machine Learning](https://www.oreilly.com/library/view/mlops-engineering/9781492083650/)
- [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)

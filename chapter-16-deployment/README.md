# Chapter 16: Deploying AI Applications

This chapter demonstrates how to deploy AI/ML models in production using Java and Spring Boot. The application provides a complete framework for model management, deployment, and serving predictions through REST APIs.

## 🎯 Learning Objectives

By the end of this chapter, you will be able to:

- Understand AI model deployment concepts and best practices
- Implement REST APIs for AI model inference using Spring Boot
- Handle model versioning, lifecycle management, and A/B testing
- Containerize AI applications using Docker
- Monitor and maintain AI systems in production
- Implement model serving architectures and scalability considerations

## 🏗️ Architecture Overview

The application follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    REST API Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ ModelController │  │PredictionController│  │Health Checks│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  ModelService   │  │PredictionService│  │Monitoring    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   AIModel       │  │ClassificationModel│  │ModelStats   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ModelRepository  │  │   H2 Database   │  │Model Storage │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Java 11 or higher
- Maven 3.6+
- Docker (optional, for containerization)

### Running the Application

#### Option 1: Local Development

1. **Clone and navigate to the chapter directory:**
   ```bash
   cd chapter-16-deployment
   ```

2. **Build the application:**
   ```bash
   mvn clean package
   ```

3. **Run the application:**
   ```bash
   java -jar target/chapter-16-deployment-1.0.0.jar
   ```

4. **Access the application:**
   - Main application: http://localhost:8080/ai-deployment
   - H2 Database Console: http://localhost:8080/ai-deployment/h2-console
   - Actuator Health: http://localhost:8080/ai-deployment/actuator/health

#### Option 2: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Main application: http://localhost:8080/ai-deployment
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## 📚 API Documentation

### Model Management APIs

#### Create a Model
```http
POST /api/v1/models
Content-Type: application/json

{
  "name": "Customer Churn Classifier",
  "version": "1.0.0",
  "description": "Logistic regression model for customer churn prediction",
  "createdBy": "data-scientist"
}
```

#### Get All Models
```http
GET /api/v1/models
```

#### Get Model by ID
```http
GET /api/v1/models/{modelId}
```

#### Deploy a Model
```http
POST /api/v1/models/{modelId}/deploy
```

#### Update Model Status
```http
PUT /api/v1/models/{modelId}/status?status=DEPLOYED
```

### Prediction APIs

#### Single Prediction
```http
POST /api/v1/predictions/{modelId}
Content-Type: application/json

{
  "features": [0.5, -0.3, 0.8, -0.2, 0.1]
}
```

#### Batch Prediction (A/B Testing)
```http
POST /api/v1/predictions/batch
Content-Type: application/json

{
  "modelIds": ["model_abc123", "model_def456"],
  "features": [0.5, -0.3, 0.8, -0.2, 0.1]
}
```

#### Get Model Statistics
```http
GET /api/v1/predictions/{modelId}/stats
```

## 🔧 Configuration

### Application Properties

Key configuration options in `application.properties`:

```properties
# Server Configuration
server.port=8080
server.servlet.context-path=/ai-deployment

# Database Configuration
spring.datasource.url=jdbc:h2:mem:ai_deployment_db

# Model Configuration
ai.model.storage.path=/tmp/ai-models
ai.model.max-file-size=100MB

# Prediction Configuration
ai.prediction.timeout-ms=5000
ai.prediction.max-batch-size=100

# Monitoring Configuration
ai.monitoring.enabled=true
ai.monitoring.metrics-interval-ms=60000
```

### Environment Variables

For Docker deployment, you can override settings:

```bash
export SPRING_PROFILES_ACTIVE=docker
export JAVA_OPTS="-Xms512m -Xmx2g"
```

## 🏗️ Model Lifecycle Management

The application supports a complete model lifecycle:

1. **DRAFT** - Model is being developed
2. **TRAINING** - Model is being trained
3. **TRAINED** - Training completed
4. **EVALUATING** - Model is being evaluated
5. **APPROVED** - Model passed evaluation
6. **DEPLOYED** - Model is serving predictions
7. **AB_TESTING** - Model is in A/B testing
8. **DEPRECATED** - Model is being phased out
9. **ARCHIVED** - Model is archived

## 📊 Monitoring and Observability

### Built-in Metrics

The application exposes metrics via Spring Boot Actuator:

- **Health Checks**: `/actuator/health`
- **Metrics**: `/actuator/metrics`
- **Prometheus**: `/actuator/prometheus`

### Key Metrics

- Model prediction latency
- Throughput (predictions per second)
- Error rates
- Memory usage
- CPU utilization

### Monitoring Stack

When using Docker Compose, the application includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards

## 🔒 Security Considerations

### Production Security

For production deployment, consider:

1. **Authentication & Authorization**
   ```properties
   spring.security.user.name=admin
   spring.security.user.password=secure_password
   spring.security.basic.enabled=true
   ```

2. **HTTPS/TLS**
   ```properties
   server.ssl.enabled=true
   server.ssl.key-store=classpath:keystore.p12
   server.ssl.key-store-password=password
   ```

3. **Input Validation**
   - All prediction inputs are validated
   - Feature count validation
   - Range and type checking

## 🚀 Deployment Strategies

### 1. Blue-Green Deployment

```bash
# Deploy new version (green)
docker-compose -f docker-compose-green.yml up -d

# Switch traffic
curl -X PUT http://load-balancer/switch-to-green

# Decommission old version (blue)
docker-compose -f docker-compose-blue.yml down
```

### 2. Canary Deployment

```bash
# Deploy canary (10% traffic)
docker-compose -f docker-compose-canary.yml up -d

# Monitor metrics
curl http://localhost:9090/api/v1/query?query=prediction_latency

# Gradually increase traffic
curl -X PUT http://load-balancer/set-canary-weight/50
```

### 3. Rolling Updates

```bash
# Update with zero downtime
docker-compose up -d --no-deps --build ai-deployment-service
```

## 🧪 Testing

### Unit Tests

```bash
mvn test
```

### Integration Tests

```bash
mvn verify
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p test-data.json \
   http://localhost:8080/ai-deployment/api/v1/predictions/model_123
```

## 📈 Performance Optimization

### JVM Tuning

```bash
# Production JVM options
export JAVA_OPTS="-Xms2g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Model Optimization

1. **Model Quantization**: Reduce model size
2. **Batch Processing**: Process multiple predictions together
3. **Caching**: Cache frequently used predictions
4. **Async Processing**: Use async endpoints for long-running predictions

## 🔍 Troubleshooting

### Common Issues

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

## 📚 Additional Resources

- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

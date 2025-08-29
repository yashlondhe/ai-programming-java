package com.aiprogramming.ch16;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Main Spring Boot application for AI model deployment
 * 
 * This application demonstrates various aspects of deploying AI models:
 * - REST API endpoints for model inference
 * - Model versioning and management
 * - Performance monitoring and logging
 * - Docker containerization support
 * - A/B testing capabilities
 */
@SpringBootApplication
@EnableAsync
@EnableScheduling
public class DeploymentApplication {

    public static void main(String[] args) {
        SpringApplication.run(DeploymentApplication.class, args);
    }
}

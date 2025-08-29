package com.aiprogramming.ch08;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Example application demonstrating CNN for image classification
 */
public class ImageClassificationExample {
    
    public static void main(String[] args) {
        System.out.println("=== CNN Image Classification Example ===");
        
        // Create synthetic dataset for demonstration
        List<Tensor> trainingImages = createSyntheticDataset(100, 28, 28);
        List<Tensor> trainingLabels = createSyntheticLabels(100, 10);
        
        List<Tensor> testImages = createSyntheticDataset(20, 28, 28);
        List<Tensor> testLabels = createSyntheticLabels(20, 10);
        
        System.out.println("Dataset created:");
        System.out.println("- Training images: " + trainingImages.size());
        System.out.println("- Test images: " + testImages.size());
        System.out.println("- Image size: 28x28 (MNIST-like)");
        System.out.println("- Number of classes: 10");
        
        // Create CNN architecture
        CNN cnn = new CNN(0.01);
        
        // Add layers (similar to LeNet-5 architecture)
        cnn.addConvLayer(6, 5, 1, 1, "valid");  // Conv1: 1x28x28 -> 6x24x24
        cnn.addMaxPoolLayer(2, 2);               // Pool1: 6x24x24 -> 6x12x12
        cnn.addConvLayer(16, 5, 6, 1, "valid"); // Conv2: 6x12x12 -> 16x8x8
        cnn.addMaxPoolLayer(2, 2);               // Pool2: 16x8x8 -> 16x4x4
        cnn.addGlobalAveragePoolLayer();         // Global Avg Pool: 16x4x4 -> 16x1x1
        cnn.addDenseLayer(16, 10, new Sigmoid()); // Dense: 16 -> 10 (output classes)
        
        System.out.println("\nCNN Architecture:");
        System.out.println("1. Conv2D: 6 filters, 5x5 kernel");
        System.out.println("2. MaxPool: 2x2, stride 2");
        System.out.println("3. Conv2D: 16 filters, 5x5 kernel");
        System.out.println("4. MaxPool: 2x2, stride 2");
        System.out.println("5. Global Average Pooling");
        System.out.println("6. Dense: 16 -> 10 (output)");
        
        // Train the network
        System.out.println("\nTraining CNN...");
        List<Double> losses = cnn.train(trainingImages, trainingLabels, 50);
        
        System.out.println("\nTraining completed!");
        System.out.println("Final loss: " + losses.get(losses.size() - 1));
        
        // Test the network
        System.out.println("\nTesting CNN...");
        double accuracy = evaluateAccuracy(cnn, testImages, testLabels);
        System.out.println("Test accuracy: " + (accuracy * 100) + "%");
        
        // Demonstrate prediction
        System.out.println("\nSample predictions:");
        for (int i = 0; i < 5; i++) {
            Tensor prediction = cnn.predict(testImages.get(i));
            int predictedClass = getPredictedClass(prediction);
            int trueClass = getPredictedClass(testLabels.get(i));
            
            System.out.printf("Sample %d: Predicted=%d, True=%d, Correct=%s%n", 
                            i + 1, predictedClass, trueClass, 
                            predictedClass == trueClass ? "Yes" : "No");
        }
        
        // Demonstrate convolution operation
        System.out.println("\n=== Convolution Operation Demo ===");
        demonstrateConvolution();
        
        // Demonstrate pooling operation
        System.out.println("\n=== Pooling Operation Demo ===");
        demonstratePooling();
        
        // Demonstrate image processing
        System.out.println("\n=== Image Processing Demo ===");
        demonstrateImageProcessing();
    }
    
    /**
     * Create synthetic image dataset
     */
    private static List<Tensor> createSyntheticDataset(int numImages, int height, int width) {
        List<Tensor> images = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < numImages; i++) {
            Tensor image = Tensor.zeros(1, height, width);
            
            // Create synthetic patterns
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // Create different patterns based on image index
                    double value;
                    if (i % 10 == 0) {
                        // Horizontal lines
                        value = (h % 5 == 0) ? 0.8 : 0.1;
                    } else if (i % 10 == 1) {
                        // Vertical lines
                        value = (w % 5 == 0) ? 0.8 : 0.1;
                    } else if (i % 10 == 2) {
                        // Diagonal pattern
                        value = ((h + w) % 8 == 0) ? 0.8 : 0.1;
                    } else if (i % 10 == 3) {
                        // Checkerboard pattern
                        value = ((h / 4 + w / 4) % 2 == 0) ? 0.8 : 0.1;
                    } else if (i % 10 == 4) {
                        // Random noise
                        value = random.nextDouble() * 0.5 + 0.2;
                    } else {
                        // Mixed patterns
                        value = random.nextDouble() * 0.6 + 0.2;
                        if (h < height / 3) value += 0.3;
                        if (w < width / 3) value += 0.2;
                    }
                    
                    image.set(Math.min(1.0, Math.max(0.0, value)), 0, h, w);
                }
            }
            
            images.add(image);
        }
        
        return images;
    }
    
    /**
     * Create synthetic labels (one-hot encoded)
     */
    private static List<Tensor> createSyntheticLabels(int numLabels, int numClasses) {
        List<Tensor> labels = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numLabels; i++) {
            int classIndex = random.nextInt(numClasses);
            Tensor label = Tensor.zeros(numClasses);
            label.set(1.0, classIndex);
            labels.add(label);
        }
        
        return labels;
    }
    
    /**
     * Evaluate accuracy of the model
     */
    private static double evaluateAccuracy(CNN cnn, List<Tensor> images, List<Tensor> labels) {
        int correct = 0;
        int total = images.size();
        
        for (int i = 0; i < total; i++) {
            Tensor prediction = cnn.predict(images.get(i));
            int predictedClass = getPredictedClass(prediction);
            int trueClass = getPredictedClass(labels.get(i));
            
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        
        return (double) correct / total;
    }
    
    /**
     * Get predicted class from output tensor
     */
    private static int getPredictedClass(Tensor output) {
        double[] data = output.getData();
        int maxIndex = 0;
        double maxValue = data[0];
        
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxValue) {
                maxValue = data[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * Demonstrate convolution operation
     */
    private static void demonstrateConvolution() {
        System.out.println("Creating a simple 3x3 image...");
        Tensor image = Tensor.zeros(1, 3, 3);
        image.set(1.0, 0, 0, 0);
        image.set(1.0, 0, 0, 1);
        image.set(1.0, 0, 0, 2);
        image.set(1.0, 0, 1, 0);
        image.set(1.0, 0, 1, 1);
        image.set(1.0, 0, 1, 2);
        image.set(1.0, 0, 2, 0);
        image.set(1.0, 0, 2, 1);
        image.set(1.0, 0, 2, 2);
        
        System.out.println("Input image:");
        printTensor(image);
        
        System.out.println("Creating a 2x2 kernel...");
        Tensor kernel = Tensor.zeros(1, 2, 2);
        kernel.set(1.0, 0, 0, 0);
        kernel.set(-1.0, 0, 0, 1);
        kernel.set(-1.0, 0, 1, 0);
        kernel.set(1.0, 0, 1, 1);
        
        System.out.println("Kernel:");
        printTensor(kernel);
        
        System.out.println("Applying convolution...");
        Tensor result = Convolution.convolve2D(image, kernel, 1, "valid");
        
        System.out.println("Convolution result:");
        printTensor(result);
    }
    
    /**
     * Demonstrate pooling operation
     */
    private static void demonstratePooling() {
        System.out.println("Creating a 4x4 image...");
        Tensor image = Tensor.zeros(1, 4, 4);
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                image.set(h * 4 + w, 0, h, w);
            }
        }
        
        System.out.println("Input image:");
        printTensor(image);
        
        System.out.println("Applying max pooling (2x2, stride 2)...");
        Tensor maxPoolResult = Pooling.maxPool(image, 2, 2);
        
        System.out.println("Max pooling result:");
        printTensor(maxPoolResult);
        
        System.out.println("Applying average pooling (2x2, stride 2)...");
        Tensor avgPoolResult = Pooling.averagePool(image, 2, 2);
        
        System.out.println("Average pooling result:");
        printTensor(avgPoolResult);
    }
    
    /**
     * Demonstrate image processing operations
     */
    private static void demonstrateImageProcessing() {
        System.out.println("Creating a test image...");
        Tensor image = Tensor.zeros(1, 4, 4);
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                image.set(0.5, 0, h, w);
            }
        }
        
        System.out.println("Original image:");
        printTensor(image);
        
        System.out.println("Normalized image:");
        Tensor normalized = ImageProcessor.normalizeImage(image);
        printTensor(normalized);
        
        System.out.println("Horizontally flipped image:");
        Tensor flipped = ImageProcessor.horizontalFlip(image);
        printTensor(flipped);
        
        System.out.println("Brightness adjusted image:");
        Tensor brightened = ImageProcessor.randomBrightness(image, 0.5);
        printTensor(brightened);
    }
    
    /**
     * Print tensor in a readable format
     */
    private static void printTensor(Tensor tensor) {
        int[] shape = tensor.getShape();
        if (shape.length == 3) {
            int channels = shape[0];
            int height = shape[1];
            int width = shape[2];
            
            for (int c = 0; c < channels; c++) {
                System.out.println("Channel " + c + ":");
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        System.out.printf("%6.2f ", tensor.get(c, h, w));
                    }
                    System.out.println();
                }
                System.out.println();
            }
        } else if (shape.length == 1) {
            System.out.print("[");
            for (int i = 0; i < shape[0]; i++) {
                System.out.printf("%.2f", tensor.get(i));
                if (i < shape[0] - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
}

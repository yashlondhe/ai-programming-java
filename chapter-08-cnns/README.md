# Chapter 8: Convolutional Neural Networks (CNNs)

This chapter implements Convolutional Neural Networks from scratch in Java, covering the fundamental concepts of CNNs, convolution operations, pooling, and image processing for computer vision tasks.

## Overview

This project provides a complete implementation of CNNs with the following features:

- **Tensor Operations**: Multi-dimensional tensor support for CNN operations
- **Convolutional Layers**: 2D convolution with customizable kernels, strides, and padding
- **Pooling Layers**: Max pooling, average pooling, and global pooling operations
- **Image Processing**: Image loading, preprocessing, augmentation, and normalization
- **CNN Architecture**: Complete CNN with convolutional and dense layers
- **Training**: Backpropagation and gradient descent for CNN training
- **Data Augmentation**: Random flips, rotations, brightness, and contrast adjustments

## Project Structure

```
src/main/java/com/aiprogramming/ch08/
├── Tensor.java                    # Multi-dimensional tensor implementation
├── Convolution.java               # 2D convolution operations
├── Pooling.java                   # Pooling operations (max, average, global)
├── CNNLayer.java                  # Abstract CNN layer and implementations
├── CNN.java                       # Main CNN class with training
├── ImageProcessor.java            # Image processing and augmentation utilities
└── ImageClassificationExample.java # Complete CNN example application
```

## Key Components

### 1. Tensor
Multi-dimensional array implementation for CNN operations with support for:
- Basic operations (add, multiply, reshape)
- Random initialization
- Shape management
- Efficient indexing

### 2. Convolution
2D convolution operations with:
- **Valid Padding**: No padding, output size reduced
- **Same Padding**: Output size matches input size
- **Customizable Stride**: Control output resolution
- **Multiple Kernels**: Support for multiple filters
- **Gradient Computation**: Backpropagation support

### 3. Pooling
Pooling operations for dimensionality reduction:
- **Max Pooling**: Selects maximum value in window
- **Average Pooling**: Computes average in window
- **Global Pooling**: Reduces spatial dimensions to 1x1
- **Gradient Computation**: Backpropagation support

### 4. CNN Layers
Layer implementations for building CNNs:
- **Conv2DLayer**: Convolutional layer with learnable kernels and biases
- **MaxPoolingLayer**: Max pooling layer
- **AveragePoolingLayer**: Average pooling layer
- **GlobalAveragePoolingLayer**: Global average pooling
- **DenseLayer**: Fully connected layer for classification

### 5. Image Processing
Comprehensive image processing utilities:
- **Image Loading**: Load images from files
- **Tensor Conversion**: Convert between images and tensors
- **Resizing**: Resize images to target dimensions
- **Normalization**: Zero-mean, unit-variance normalization
- **Data Augmentation**: Random flips, rotations, brightness, contrast
- **Batch Processing**: Create batches for training

## Usage Examples

### Basic CNN Architecture
```java
// Create CNN with learning rate
CNN cnn = new CNN(0.01);

// Add convolutional layers
cnn.addConvLayer(6, 5, 1, 1, "valid");  // 6 filters, 5x5 kernel
cnn.addMaxPoolLayer(2, 2);               // 2x2 max pooling
cnn.addConvLayer(16, 5, 6, 1, "valid"); // 16 filters, 5x5 kernel
cnn.addMaxPoolLayer(2, 2);               // 2x2 max pooling
cnn.addGlobalAveragePoolLayer();         // Global average pooling
cnn.addDenseLayer(16, 10, new Sigmoid()); // Output layer
```

### Image Processing
```java
// Load and preprocess image
Tensor image = ImageProcessor.loadImage("image.jpg");
image = ImageProcessor.resizeImage(image, 28, 28);
image = ImageProcessor.normalizeImage(image);

// Apply data augmentation
Tensor augmented = ImageProcessor.augmentImage(
    image, 0.5, 0.1, 0.2, 0.2  // flip, rotation, brightness, contrast
);
```

### Training CNN
```java
// Prepare training data
List<Tensor> images = loadTrainingImages();
List<Tensor> labels = createOneHotLabels();

// Train the network
List<Double> losses = cnn.train(images, labels, 100);

// Make predictions
Tensor prediction = cnn.predict(testImage);
int predictedClass = getPredictedClass(prediction);
```

### Convolution Operation
```java
// Create input tensor (1 channel, 3x3 image)
Tensor input = Tensor.zeros(1, 3, 3);
// Set pixel values...

// Create kernel (1 channel, 2x2 kernel)
Tensor kernel = Tensor.zeros(1, 2, 2);
// Set kernel values...

// Apply convolution
Tensor output = Convolution.convolve2D(input, kernel, 1, "valid");
```

### Pooling Operation
```java
// Apply max pooling
Tensor pooled = Pooling.maxPool(input, 2, 2); // 2x2 window, stride 2

// Apply global average pooling
Tensor globalPooled = Pooling.globalAveragePool(input);
```

## Building and Running

### Prerequisites
- Java 11 or higher
- Maven 3.6 or higher

### Build the Project
```bash
mvn clean compile
```

### Run the Example
```bash
# Run the main CNN example
mvn exec:java -Dexec.mainClass="com.aiprogramming.ch08.ImageClassificationExample"
```

### Run Tests
```bash
mvn test
```

## Key Concepts Covered

1. **Convolution**: How convolutional layers extract features from images
2. **Pooling**: Dimensionality reduction and feature selection
3. **Tensor Operations**: Multi-dimensional array manipulation
4. **Backpropagation**: Gradient computation for CNN training
5. **Image Processing**: Preprocessing and augmentation techniques
6. **CNN Architecture**: Layer composition and network design
7. **Feature Extraction**: How CNNs learn hierarchical features

## Learning Objectives

By working with this code, you will understand:

- How convolutional layers work and extract features
- The role of pooling layers in CNNs
- How to implement backpropagation for CNNs
- Image preprocessing and augmentation techniques
- CNN architecture design principles
- Tensor operations for deep learning
- Training CNNs from scratch

## CNN Architecture Details

### LeNet-5 Inspired Architecture
The example implements a simplified LeNet-5 architecture:
1. **Conv2D**: 6 filters, 5x5 kernel → Feature extraction
2. **MaxPool**: 2x2, stride 2 → Dimensionality reduction
3. **Conv2D**: 16 filters, 5x5 kernel → More complex features
4. **MaxPool**: 2x2, stride 2 → Further reduction
5. **Global Avg Pool**: Spatial to feature vector
6. **Dense**: 16 → 10 → Classification

### Feature Learning Process
1. **Low-level features**: Edges, lines, textures
2. **Mid-level features**: Shapes, patterns
3. **High-level features**: Objects, structures
4. **Classification**: Final prediction

## Exercises

The chapter includes several exercises to reinforce learning:

1. **Basic CNN**: Implement a simple CNN for digit classification
2. **Architecture Design**: Experiment with different CNN architectures
3. **Kernel Visualization**: Visualize learned convolutional kernels
4. **Data Augmentation**: Implement custom augmentation techniques
5. **Transfer Learning**: Adapt pre-trained CNNs for new tasks
6. **Object Detection**: Extend CNNs for object detection
7. **Image Segmentation**: Implement semantic segmentation
8. **Real-world Applications**: Apply CNNs to real image datasets

## Advanced Topics

### Transfer Learning
- Use pre-trained models (VGG, ResNet, etc.)
- Fine-tuning for specific tasks
- Feature extraction from intermediate layers

### Object Detection
- Region proposal networks
- Bounding box regression
- Multi-scale detection

### Image Segmentation
- Fully convolutional networks
- U-Net architecture
- Pixel-wise classification

### Optimization Techniques
- Batch normalization
- Dropout regularization
- Learning rate scheduling
- Advanced optimizers (Adam, RMSprop)

## Performance Considerations

### Memory Management
- Efficient tensor operations
- Batch processing
- Gradient accumulation

### Computational Optimization
- Parallel processing
- GPU acceleration
- Optimized convolution algorithms

### Training Strategies
- Mini-batch training
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Next Steps

After mastering this chapter, you can explore:

- Advanced CNN architectures (ResNet, DenseNet, etc.)
- Object detection frameworks (YOLO, Faster R-CNN)
- Image segmentation (U-Net, DeepLab)
- Attention mechanisms and transformers
- Computer vision applications
- Real-world deployment strategies

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or image resolution
2. **Training Instability**: Adjust learning rate or use normalization
3. **Overfitting**: Add regularization or more training data
4. **Poor Performance**: Check data preprocessing and model architecture

### Debugging Tips
- Visualize intermediate layer outputs
- Monitor gradient norms during training
- Check tensor shapes at each layer
- Validate convolution and pooling operations

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Add new features
- Improve documentation
- Create additional examples

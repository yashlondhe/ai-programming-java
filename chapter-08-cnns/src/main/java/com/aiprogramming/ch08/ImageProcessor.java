package com.aiprogramming.ch08;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;
import java.util.List;

/**
 * Image processing utilities for CNN training
 */
public class ImageProcessor {
    
    /**
     * Load image from file and convert to tensor
     */
    public static Tensor loadImage(String filePath) throws IOException {
        BufferedImage image = ImageIO.read(new File(filePath));
        return imageToTensor(image);
    }
    
    /**
     * Convert BufferedImage to tensor
     */
    public static Tensor imageToTensor(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Convert to grayscale if needed
        if (image.getType() != BufferedImage.TYPE_BYTE_GRAY) {
            BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            grayImage.getGraphics().drawImage(image, 0, 0, null);
            image = grayImage;
        }
        
        // Get pixel data
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        
        // Create tensor (channels, height, width)
        Tensor tensor = Tensor.zeros(1, height, width);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = h * width + w;
                // Convert byte to double and normalize to [0, 1]
                double pixelValue = (pixels[index] & 0xFF) / 255.0;
                tensor.set(pixelValue, 0, h, w);
            }
        }
        
        return tensor;
    }
    
    /**
     * Convert tensor back to BufferedImage
     */
    public static BufferedImage tensorToImage(Tensor tensor) {
        int[] shape = tensor.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("Tensor must be 3D (channels, height, width)");
        }
        
        int channels = shape[0];
        int height = shape[1];
        int width = shape[2];
        
        BufferedImage image;
        if (channels == 1) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        } else if (channels == 3) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        } else {
            throw new IllegalArgumentException("Unsupported number of channels: " + channels);
        }
        
        // Convert tensor data to image
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (channels == 1) {
                    double pixelValue = tensor.get(0, h, w);
                    int grayValue = (int) Math.max(0, Math.min(255, pixelValue * 255));
                    int rgb = (grayValue << 16) | (grayValue << 8) | grayValue;
                    image.setRGB(w, h, rgb);
                } else if (channels == 3) {
                    double r = tensor.get(0, h, w);
                    double g = tensor.get(1, h, w);
                    double b = tensor.get(2, h, w);
                    
                    int red = (int) Math.max(0, Math.min(255, r * 255));
                    int green = (int) Math.max(0, Math.min(255, g * 255));
                    int blue = (int) Math.max(0, Math.min(255, b * 255));
                    
                    int rgb = (red << 16) | (green << 8) | blue;
                    image.setRGB(w, h, rgb);
                }
            }
        }
        
        return image;
    }
    
    /**
     * Resize image to specified dimensions
     */
    public static Tensor resizeImage(Tensor image, int newHeight, int newWidth) {
        int[] shape = image.getShape();
        int channels = shape[0];
        int height = shape[1];
        int width = shape[2];
        
        Tensor resized = Tensor.zeros(channels, newHeight, newWidth);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < newHeight; h++) {
                for (int w = 0; w < newWidth; w++) {
                    // Simple nearest neighbor interpolation
                    int srcH = (int) (h * (double) height / newHeight);
                    int srcW = (int) (w * (double) width / newWidth);
                    
                    srcH = Math.min(srcH, height - 1);
                    srcW = Math.min(srcW, width - 1);
                    
                    double pixelValue = image.get(c, srcH, srcW);
                    resized.set(pixelValue, c, h, w);
                }
            }
        }
        
        return resized;
    }
    
    /**
     * Normalize image to zero mean and unit variance
     */
    public static Tensor normalizeImage(Tensor image) {
        int[] shape = image.getShape();
        int totalSize = image.getTotalSize();
        double[] data = image.getData();
        
        // Calculate mean
        double sum = 0.0;
        for (int i = 0; i < totalSize; i++) {
            sum += data[i];
        }
        double mean = sum / totalSize;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (int i = 0; i < totalSize; i++) {
            double diff = data[i] - mean;
            variance += diff * diff;
        }
        double std = Math.sqrt(variance / totalSize);
        
        // Normalize
        Tensor normalized = new Tensor(shape);
        double[] normalizedData = normalized.getData();
        for (int i = 0; i < totalSize; i++) {
            normalizedData[i] = (data[i] - mean) / (std + 1e-8); // Add small epsilon to avoid division by zero
        }
        
        return normalized;
    }
    
    /**
     * Apply random horizontal flip for data augmentation
     */
    public static Tensor randomHorizontalFlip(Tensor image, double probability) {
        Random random = new Random();
        if (random.nextDouble() < probability) {
            return horizontalFlip(image);
        }
        return image;
    }
    
    /**
     * Apply horizontal flip to image
     */
    public static Tensor horizontalFlip(Tensor image) {
        int[] shape = image.getShape();
        int channels = shape[0];
        int height = shape[1];
        int width = shape[2];
        
        Tensor flipped = Tensor.zeros(channels, height, width);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double pixelValue = image.get(c, h, w);
                    flipped.set(pixelValue, c, h, width - 1 - w);
                }
            }
        }
        
        return flipped;
    }
    
    /**
     * Apply random rotation for data augmentation
     */
    public static Tensor randomRotation(Tensor image, double maxAngle) {
        Random random = new Random();
        double angle = (random.nextDouble() - 0.5) * 2 * maxAngle; // [-maxAngle, maxAngle]
        return rotateImage(image, angle);
    }
    
    /**
     * Rotate image by specified angle (simplified implementation)
     */
    public static Tensor rotateImage(Tensor image, double angle) {
        int[] shape = image.getShape();
        int channels = shape[0];
        int height = shape[1];
        int width = shape[2];
        
        Tensor rotated = Tensor.zeros(channels, height, width);
        
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        
        int centerH = height / 2;
        int centerW = width / 2;
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // Calculate source coordinates
                    int srcH = (int) (cos * (h - centerH) - sin * (w - centerW) + centerH);
                    int srcW = (int) (sin * (h - centerH) + cos * (w - centerW) + centerW);
                    
                    // Check bounds
                    if (srcH >= 0 && srcH < height && srcW >= 0 && srcW < width) {
                        double pixelValue = image.get(c, srcH, srcW);
                        rotated.set(pixelValue, c, h, w);
                    }
                }
            }
        }
        
        return rotated;
    }
    
    /**
     * Apply random brightness adjustment
     */
    public static Tensor randomBrightness(Tensor image, double factor) {
        Random random = new Random();
        double brightness = 1.0 + (random.nextDouble() - 0.5) * 2 * factor; // [1-factor, 1+factor]
        
        int[] shape = image.getShape();
        int totalSize = image.getTotalSize();
        double[] data = image.getData();
        
        Tensor adjusted = new Tensor(shape);
        double[] adjustedData = adjusted.getData();
        
        for (int i = 0; i < totalSize; i++) {
            adjustedData[i] = Math.max(0.0, Math.min(1.0, data[i] * brightness));
        }
        
        return adjusted;
    }
    
    /**
     * Apply random contrast adjustment
     */
    public static Tensor randomContrast(Tensor image, double factor) {
        Random random = new Random();
        double contrast = 1.0 + (random.nextDouble() - 0.5) * 2 * factor; // [1-factor, 1+factor]
        
        int[] shape = image.getShape();
        int totalSize = image.getTotalSize();
        double[] data = image.getData();
        
        // Calculate mean
        double sum = 0.0;
        for (int i = 0; i < totalSize; i++) {
            sum += data[i];
        }
        double mean = sum / totalSize;
        
        Tensor adjusted = new Tensor(shape);
        double[] adjustedData = adjusted.getData();
        
        for (int i = 0; i < totalSize; i++) {
            double adjustedValue = (data[i] - mean) * contrast + mean;
            adjustedData[i] = Math.max(0.0, Math.min(1.0, adjustedValue));
        }
        
        return adjusted;
    }
    
    /**
     * Apply multiple augmentations to an image
     */
    public static Tensor augmentImage(Tensor image, double flipProb, double maxRotation, 
                                    double brightnessFactor, double contrastFactor) {
        Tensor augmented = image;
        
        // Apply augmentations
        augmented = randomHorizontalFlip(augmented, flipProb);
        augmented = randomRotation(augmented, maxRotation);
        augmented = randomBrightness(augmented, brightnessFactor);
        augmented = randomContrast(augmented, contrastFactor);
        
        return augmented;
    }
    
    /**
     * Create a batch of images from a list of tensors
     */
    public static Tensor createBatch(List<Tensor> images) {
        if (images.isEmpty()) {
            throw new IllegalArgumentException("Cannot create batch from empty list");
        }
        
        int batchSize = images.size();
        int[] firstShape = images.get(0).getShape();
        
        // Check that all images have the same shape
        for (Tensor image : images) {
            if (!java.util.Arrays.equals(image.getShape(), firstShape)) {
                throw new IllegalArgumentException("All images must have the same shape");
            }
        }
        
        // Create batch tensor (batch_size, channels, height, width)
        int[] batchShape = new int[firstShape.length + 1];
        batchShape[0] = batchSize;
        for (int i = 0; i < firstShape.length; i++) {
            batchShape[i + 1] = firstShape[i];
        }
        
        Tensor batch = new Tensor(batchShape);
        int imageSize = images.get(0).getTotalSize();
        
        for (int i = 0; i < batchSize; i++) {
            double[] imageData = images.get(i).getData();
            for (int j = 0; j < imageSize; j++) {
                int batchIndex = i * imageSize + j;
                batch.getData()[batchIndex] = imageData[j];
            }
        }
        
        return batch;
    }
    
    /**
     * Save tensor as image to file
     */
    public static void saveImage(Tensor tensor, String filePath) throws IOException {
        BufferedImage image = tensorToImage(tensor);
        ImageIO.write(image, "png", new File(filePath));
    }
}

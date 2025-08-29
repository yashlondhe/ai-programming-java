package com.aiprogramming.ch13;

import java.util.List;

/**
 * Comprehensive example demonstrating computer vision algorithms
 * Shows image processing, edge detection, and feature detection techniques
 */
public class ComputerVisionExample {
    
    public static void main(String[] args) {
        System.out.println("=== Computer Vision Algorithms Demonstration ===\n");
        
        // Create a synthetic test image
        Image testImage = createTestImage();
        System.out.println("Created test image: " + testImage);
        
        // Demonstrate image processing operations
        demonstrateImageProcessing(testImage);
        
        // Demonstrate edge detection
        demonstrateEdgeDetection(testImage);
        
        // Demonstrate feature detection
        demonstrateFeatureDetection(testImage);
        
        // Demonstrate advanced operations
        demonstrateAdvancedOperations(testImage);
        
        System.out.println("\n=== Computer Vision Demonstration Complete ===");
    }
    
    /**
     * Create a synthetic test image with various features
     */
    private static Image createTestImage() {
        Image image = Image.createGrayscale(256, 256);
        
        // Create a simple pattern with edges and corners
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double value = 128.0;
                
                // Add some geometric shapes
                if (x < 64 || x > 192 || y < 64 || y > 192) {
                    value = 64.0; // Dark border
                }
                
                // Add a circle
                double centerX = 128.0;
                double centerY = 128.0;
                double radius = 40.0;
                double distance = Math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                if (distance < radius) {
                    value = 200.0; // Bright circle
                }
                
                // Add some noise
                value += (Math.random() - 0.5) * 20.0;
                
                image.setGrayscalePixel(x, y, Math.max(0, Math.min(255, value)));
            }
        }
        
        return image;
    }
    
    /**
     * Demonstrate basic image processing operations
     */
    private static void demonstrateImageProcessing(Image image) {
        System.out.println("=== Image Processing Operations ===");
        
        // Gaussian blur
        System.out.println("Applying Gaussian blur...");
        Image blurred = ImageProcessing.gaussianBlur(image, 2.0);
        System.out.println("Gaussian blur applied with sigma=2.0");
        
        // Mean filter
        System.out.println("Applying mean filter...");
        Image meanFiltered = ImageProcessing.meanFilter(image, 5);
        System.out.println("Mean filter applied with 5x5 kernel");
        
        // Median filter
        System.out.println("Applying median filter...");
        Image medianFiltered = ImageProcessing.medianFilter(image, 5);
        System.out.println("Median filter applied with 5x5 kernel");
        
        // Bilateral filter
        System.out.println("Applying bilateral filter...");
        Image bilateralFiltered = ImageProcessing.bilateralFilter(image, 5.0, 50.0);
        System.out.println("Bilateral filter applied with sigmaSpace=5.0, sigmaColor=50.0");
        
        // Histogram equalization
        System.out.println("Applying histogram equalization...");
        Image equalized = ImageProcessing.histogramEqualization(image);
        System.out.println("Histogram equalization applied");
        
        // Gamma correction
        System.out.println("Applying gamma correction...");
        Image gammaCorrected = ImageProcessing.gammaCorrection(image, 1.5);
        System.out.println("Gamma correction applied with gamma=1.5");
        
        System.out.println();
    }
    
    /**
     * Demonstrate edge detection algorithms
     */
    private static void demonstrateEdgeDetection(Image image) {
        System.out.println("=== Edge Detection Algorithms ===");
        
        // Sobel edge detection
        System.out.println("Applying Sobel edge detection...");
        Image sobelEdges = EdgeDetection.sobel(image);
        System.out.println("Sobel edges detected");
        
        // Sobel with direction
        System.out.println("Computing Sobel with direction...");
        Image[] sobelWithDir = EdgeDetection.sobelWithDirection(image);
        System.out.println("Sobel magnitude and direction computed");
        
        // Laplacian edge detection
        System.out.println("Applying Laplacian edge detection...");
        Image laplacianEdges = EdgeDetection.laplacian(image);
        System.out.println("Laplacian edges detected");
        
        // Canny edge detection
        System.out.println("Applying Canny edge detection...");
        Image cannyEdges = EdgeDetection.canny(image, 50.0, 150.0);
        System.out.println("Canny edges detected with thresholds (50, 150)");
        
        // Prewitt edge detection
        System.out.println("Applying Prewitt edge detection...");
        Image prewittEdges = EdgeDetection.prewitt(image);
        System.out.println("Prewitt edges detected");
        
        // Roberts edge detection
        System.out.println("Applying Roberts edge detection...");
        Image robertsEdges = EdgeDetection.roberts(image);
        System.out.println("Roberts edges detected");
        
        System.out.println();
    }
    
    /**
     * Demonstrate feature detection algorithms
     */
    private static void demonstrateFeatureDetection(Image image) {
        System.out.println("=== Feature Detection Algorithms ===");
        
        // Harris corner detection
        System.out.println("Detecting Harris corners...");
        List<FeatureDetection.Point> harrisCorners = FeatureDetection.harrisCorners(image, 0.04, 1000.0);
        System.out.println("Harris corners detected: " + harrisCorners.size());
        if (!harrisCorners.isEmpty()) {
            System.out.println("Top 5 Harris corners:");
            for (int i = 0; i < Math.min(5, harrisCorners.size()); i++) {
                System.out.println("  " + harrisCorners.get(i));
            }
        }
        
        // Blob detection
        System.out.println("Detecting blobs...");
        List<FeatureDetection.Blob> blobs = FeatureDetection.detectBlobs(image, 1.0, 5.0, 5, 100.0);
        System.out.println("Blobs detected: " + blobs.size());
        if (!blobs.isEmpty()) {
            System.out.println("Top 5 blobs:");
            for (int i = 0; i < Math.min(5, blobs.size()); i++) {
                System.out.println("  " + blobs.get(i));
            }
        }
        
        // FAST corner detection
        System.out.println("Detecting FAST corners...");
        List<FeatureDetection.Point> fastCorners = FeatureDetection.fastCorners(image, 20, 9);
        System.out.println("FAST corners detected: " + fastCorners.size());
        if (!fastCorners.isEmpty()) {
            System.out.println("Top 5 FAST corners:");
            for (int i = 0; i < Math.min(5, fastCorners.size()); i++) {
                System.out.println("  " + fastCorners.get(i));
            }
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrate advanced operations
     */
    private static void demonstrateAdvancedOperations(Image image) {
        System.out.println("=== Advanced Operations ===");
        
        // Morphological operations
        System.out.println("Applying morphological operations...");
        
        // Create binary image for morphological operations
        Image binary = ImageProcessing.threshold(image, 128.0);
        
        // Erosion
        Image eroded = ImageProcessing.erode(binary, 3);
        System.out.println("Erosion applied with 3x3 kernel");
        
        // Dilation
        Image dilated = ImageProcessing.dilate(binary, 3);
        System.out.println("Dilation applied with 3x3 kernel");
        
        // Opening
        Image opened = ImageProcessing.open(binary, 3);
        System.out.println("Opening applied with 3x3 kernel");
        
        // Closing
        Image closed = ImageProcessing.close(binary, 3);
        System.out.println("Closing applied with 3x3 kernel");
        
        // Adaptive thresholding
        System.out.println("Applying adaptive thresholding...");
        Image adaptiveThresholded = ImageProcessing.adaptiveThreshold(image, 15, 2.0);
        System.out.println("Adaptive thresholding applied with blockSize=15, constant=2.0");
        
        // Contrast stretching
        System.out.println("Applying contrast stretching...");
        Image stretched = ImageProcessing.contrastStretching(image, 0.0, 255.0);
        System.out.println("Contrast stretching applied to full range");
        
        // Convolution with custom kernel
        System.out.println("Applying custom convolution kernel...");
        double[][] customKernel = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
        };
        Image sharpened = Convolution.convolve(image, customKernel);
        System.out.println("Sharpening kernel applied");
        
        System.out.println();
    }
    
    /**
     * Utility method to print image statistics
     */
    public static void printImageStats(Image image, String name) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        int count = 0;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double value = image.getPixel(x, y, c);
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                    sum += value;
                    count++;
                }
            }
        }
        
        double mean = sum / count;
        System.out.printf("%s: min=%.2f, max=%.2f, mean=%.2f%n", name, min, max, mean);
    }
    
    /**
     * Utility method to compare two images
     */
    public static double computeMSE(Image image1, Image image2) {
        if (image1.getWidth() != image2.getWidth() || 
            image1.getHeight() != image2.getHeight() ||
            image1.getChannels() != image2.getChannels()) {
            throw new IllegalArgumentException("Images must have same dimensions and channels");
        }
        
        double mse = 0.0;
        int count = 0;
        
        for (int y = 0; y < image1.getHeight(); y++) {
            for (int x = 0; x < image1.getWidth(); x++) {
                for (int c = 0; c < image1.getChannels(); c++) {
                    double diff = image1.getPixel(x, y, c) - image2.getPixel(x, y, c);
                    mse += diff * diff;
                    count++;
                }
            }
        }
        
        return mse / count;
    }
}

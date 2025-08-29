package com.aiprogramming.ch13;

import java.util.*;

/**
 * Implements various edge detection algorithms
 * Includes Sobel, Canny, and Laplacian edge detection methods
 */
public class EdgeDetection {
    
    /**
     * Apply Sobel edge detection
     * @param image input grayscale image
     * @return edge magnitude image
     */
    public static Image sobel(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        double[][][] sobelKernels = Convolution.createSobelKernels();
        double[][] sobelX = sobelKernels[0];
        double[][] sobelY = sobelKernels[1];
        
        // Apply Sobel kernels
        Image gradX = Convolution.convolve(image, sobelX);
        Image gradY = Convolution.convolve(image, sobelY);
        
        // Compute magnitude
        Image magnitude = new Image(image.getWidth(), image.getHeight(), 1);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double gx = gradX.getGrayscalePixel(x, y);
                double gy = gradY.getGrayscalePixel(x, y);
                double mag = Math.sqrt(gx * gx + gy * gy);
                magnitude.setGrayscalePixel(x, y, mag);
            }
        }
        
        return magnitude;
    }
    
    /**
     * Apply Sobel edge detection with direction
     * @param image input grayscale image
     * @return array containing [magnitude, direction] images
     */
    public static Image[] sobelWithDirection(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        double[][][] sobelKernels = Convolution.createSobelKernels();
        double[][] sobelX = sobelKernels[0];
        double[][] sobelY = sobelKernels[1];
        
        // Apply Sobel kernels
        Image gradX = Convolution.convolve(image, sobelX);
        Image gradY = Convolution.convolve(image, sobelY);
        
        // Compute magnitude and direction
        Image magnitude = new Image(image.getWidth(), image.getHeight(), 1);
        Image direction = new Image(image.getWidth(), image.getHeight(), 1);
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double gx = gradX.getGrayscalePixel(x, y);
                double gy = gradY.getGrayscalePixel(x, y);
                
                // Magnitude
                double mag = Math.sqrt(gx * gx + gy * gy);
                magnitude.setGrayscalePixel(x, y, mag);
                
                // Direction (in radians)
                double dir = Math.atan2(gy, gx);
                direction.setGrayscalePixel(x, y, dir);
            }
        }
        
        return new Image[]{magnitude, direction};
    }
    
    /**
     * Apply Laplacian edge detection
     * @param image input grayscale image
     * @return edge image
     */
    public static Image laplacian(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        double[][] laplacianKernel = Convolution.createLaplacianKernel();
        return Convolution.convolve(image, laplacianKernel);
    }
    
    /**
     * Apply Canny edge detection
     * @param image input grayscale image
     * @param lowThreshold low threshold for hysteresis
     * @param highThreshold high threshold for hysteresis
     * @return binary edge image
     */
    public static Image canny(Image image, double lowThreshold, double highThreshold) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        // Step 1: Gaussian smoothing
        double[][] gaussianKernel = Convolution.createGaussianKernel(5, 1.0);
        Image smoothed = Convolution.convolve(image, gaussianKernel);
        
        // Step 2: Compute gradients using Sobel
        Image[] sobelResult = sobelWithDirection(smoothed);
        Image magnitude = sobelResult[0];
        Image direction = sobelResult[1];
        
        // Step 3: Non-maximum suppression
        Image suppressed = nonMaximumSuppression(magnitude, direction);
        
        // Step 4: Hysteresis thresholding
        return hysteresisThresholding(suppressed, lowThreshold, highThreshold);
    }
    
    /**
     * Apply non-maximum suppression to gradient magnitude
     * @param magnitude gradient magnitude image
     * @param direction gradient direction image
     * @return suppressed image
     */
    private static Image nonMaximumSuppression(Image magnitude, Image direction) {
        Image result = new Image(magnitude.getWidth(), magnitude.getHeight(), 1);
        
        for (int y = 1; y < magnitude.getHeight() - 1; y++) {
            for (int x = 1; x < magnitude.getWidth() - 1; x++) {
                double mag = magnitude.getGrayscalePixel(x, y);
                double dir = direction.getGrayscalePixel(x, y);
                
                // Convert direction to degrees and normalize to [0, 180]
                double angle = Math.toDegrees(dir);
                if (angle < 0) angle += 180;
                
                // Determine gradient direction (0°, 45°, 90°, 135°)
                int gradientDir;
                if (angle < 22.5 || angle >= 157.5) {
                    gradientDir = 0; // 0° - horizontal
                } else if (angle < 67.5) {
                    gradientDir = 1; // 45°
                } else if (angle < 112.5) {
                    gradientDir = 2; // 90° - vertical
                } else {
                    gradientDir = 3; // 135°
                }
                
                // Get neighboring pixels based on gradient direction
                double neighbor1, neighbor2;
                switch (gradientDir) {
                    case 0: // Horizontal
                        neighbor1 = magnitude.getGrayscalePixel(x - 1, y);
                        neighbor2 = magnitude.getGrayscalePixel(x + 1, y);
                        break;
                    case 1: // 45°
                        neighbor1 = magnitude.getGrayscalePixel(x - 1, y - 1);
                        neighbor2 = magnitude.getGrayscalePixel(x + 1, y + 1);
                        break;
                    case 2: // Vertical
                        neighbor1 = magnitude.getGrayscalePixel(x, y - 1);
                        neighbor2 = magnitude.getGrayscalePixel(x, y + 1);
                        break;
                    case 3: // 135°
                        neighbor1 = magnitude.getGrayscalePixel(x - 1, y + 1);
                        neighbor2 = magnitude.getGrayscalePixel(x + 1, y - 1);
                        break;
                    default:
                        neighbor1 = neighbor2 = 0;
                }
                
                // Suppress if current pixel is not a local maximum
                if (mag >= neighbor1 && mag >= neighbor2) {
                    result.setGrayscalePixel(x, y, mag);
                } else {
                    result.setGrayscalePixel(x, y, 0);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Apply hysteresis thresholding
     * @param image input image
     * @param lowThreshold low threshold
     * @param highThreshold high threshold
     * @return binary edge image
     */
    private static Image hysteresisThresholding(Image image, double lowThreshold, double highThreshold) {
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        boolean[][] visited = new boolean[image.getHeight()][image.getWidth()];
        
        // First pass: mark strong edges
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double pixel = image.getGrayscalePixel(x, y);
                if (pixel >= highThreshold) {
                    result.setGrayscalePixel(x, y, 255);
                    visited[y][x] = true;
                } else if (pixel < lowThreshold) {
                    result.setGrayscalePixel(x, y, 0);
                    visited[y][x] = true;
                }
            }
        }
        
        // Second pass: connect weak edges to strong edges
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                if (!visited[y][x]) {
                    double pixel = image.getGrayscalePixel(x, y);
                    if (pixel >= lowThreshold) {
                        // Check if connected to a strong edge
                        if (isConnectedToStrongEdge(result, x, y)) {
                            result.setGrayscalePixel(x, y, 255);
                        } else {
                            result.setGrayscalePixel(x, y, 0);
                        }
                    } else {
                        result.setGrayscalePixel(x, y, 0);
                    }
                    visited[y][x] = true;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Check if a pixel is connected to a strong edge
     */
    private static boolean isConnectedToStrongEdge(Image image, int x, int y) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < image.getWidth() && ny >= 0 && ny < image.getHeight()) {
                    if (image.getGrayscalePixel(nx, ny) == 255) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    /**
     * Apply Prewitt edge detection
     * @param image input grayscale image
     * @return edge magnitude image
     */
    public static Image prewitt(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        double[][] prewittX = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        
        double[][] prewittY = {
            {-1, -1, -1},
            { 0,  0,  0},
            { 1,  1,  1}
        };
        
        // Apply Prewitt kernels
        Image gradX = Convolution.convolve(image, prewittX);
        Image gradY = Convolution.convolve(image, prewittY);
        
        // Compute magnitude
        Image magnitude = new Image(image.getWidth(), image.getHeight(), 1);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double gx = gradX.getGrayscalePixel(x, y);
                double gy = gradY.getGrayscalePixel(x, y);
                double mag = Math.sqrt(gx * gx + gy * gy);
                magnitude.setGrayscalePixel(x, y, mag);
            }
        }
        
        return magnitude;
    }
    
    /**
     * Apply Roberts cross edge detection
     * @param image input grayscale image
     * @return edge magnitude image
     */
    public static Image roberts(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        double[][] robertsX = {
            {1,  0},
            {0, -1}
        };
        
        double[][] robertsY = {
            { 0, 1},
            {-1, 0}
        };
        
        // Apply Roberts kernels
        Image gradX = Convolution.convolve(image, robertsX);
        Image gradY = Convolution.convolve(image, robertsY);
        
        // Compute magnitude
        Image magnitude = new Image(image.getWidth(), image.getHeight(), 1);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double gx = gradX.getGrayscalePixel(x, y);
                double gy = gradY.getGrayscalePixel(x, y);
                double mag = Math.sqrt(gx * gx + gy * gy);
                magnitude.setGrayscalePixel(x, y, mag);
            }
        }
        
        return magnitude;
    }
}

package com.aiprogramming.ch13;

import java.util.*;

/**
 * Implements feature detection algorithms
 * Includes corner detection, blob detection, and interest point detection
 */
public class FeatureDetection {
    
    /**
     * Harris corner detection
     * @param image input grayscale image
     * @param k Harris parameter (typically 0.04-0.06)
     * @param threshold corner response threshold
     * @return list of corner points
     */
    public static List<Point> harrisCorners(Image image, double k, double threshold) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        // Compute gradients
        double[][][] sobelKernels = Convolution.createSobelKernels();
        Image gradX = Convolution.convolve(image, sobelKernels[0]);
        Image gradY = Convolution.convolve(image, sobelKernels[1]);
        
        // Compute Harris response
        Image harrisResponse = computeHarrisResponse(gradX, gradY, k);
        
        // Find local maxima
        return findLocalMaxima(harrisResponse, threshold);
    }
    
    /**
     * Compute Harris corner response
     */
    private static Image computeHarrisResponse(Image gradX, Image gradY, double k) {
        int width = gradX.getWidth();
        int height = gradX.getHeight();
        Image response = new Image(width, height, 1);
        
        // Gaussian kernel for smoothing
        double[][] gaussianKernel = Convolution.createGaussianKernel(5, 1.0);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gx = gradX.getGrayscalePixel(x, y);
                double gy = gradY.getGrayscalePixel(x, y);
                
                // Compute structure tensor elements
                double Ixx = gx * gx;
                double Iyy = gy * gy;
                double Ixy = gx * gy;
                
                // Apply Gaussian smoothing to structure tensor
                double sumIxx = 0, sumIyy = 0, sumIxy = 0;
                double kernelSum = 0;
                
                for (int ky = 0; ky < gaussianKernel.length; ky++) {
                    for (int kx = 0; kx < gaussianKernel[0].length; kx++) {
                        int nx = x + kx - gaussianKernel[0].length / 2;
                        int ny = y + ky - gaussianKernel.length / 2;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            double weight = gaussianKernel[ky][kx];
                            sumIxx += weight * Ixx;
                            sumIyy += weight * Iyy;
                            sumIxy += weight * Ixy;
                            kernelSum += weight;
                        }
                    }
                }
                
                // Normalize
                sumIxx /= kernelSum;
                sumIyy /= kernelSum;
                sumIxy /= kernelSum;
                
                // Compute Harris response: det(M) - k * trace(M)^2
                double det = sumIxx * sumIyy - sumIxy * sumIxy;
                double trace = sumIxx + sumIyy;
                double harrisValue = det - k * trace * trace;
                
                response.setGrayscalePixel(x, y, harrisValue);
            }
        }
        
        return response;
    }
    
    /**
     * Find local maxima in an image
     */
    private static List<Point> findLocalMaxima(Image image, double threshold) {
        List<Point> maxima = new ArrayList<>();
        int width = image.getWidth();
        int height = image.getHeight();
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double center = image.getGrayscalePixel(x, y);
                
                if (center > threshold) {
                    boolean isMax = true;
                    
                    // Check 8-neighborhood
                    for (int dy = -1; dy <= 1 && isMax; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            
                            double neighbor = image.getGrayscalePixel(x + dx, y + dy);
                            if (neighbor >= center) {
                                isMax = false;
                                break;
                            }
                        }
                    }
                    
                    if (isMax) {
                        maxima.add(new Point(x, y, center));
                    }
                }
            }
        }
        
        // Sort by response strength
        maxima.sort((a, b) -> Double.compare(b.response, a.response));
        return maxima;
    }
    
    /**
     * Blob detection using Laplacian of Gaussian
     * @param image input grayscale image
     * @param minSigma minimum scale
     * @param maxSigma maximum scale
     * @param numScales number of scales to test
     * @param threshold blob response threshold
     * @return list of detected blobs
     */
    public static List<Blob> detectBlobs(Image image, double minSigma, double maxSigma, 
                                       int numScales, double threshold) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        List<Blob> blobs = new ArrayList<>();
        
        // Test different scales
        for (int i = 0; i < numScales; i++) {
            double sigma = minSigma + (maxSigma - minSigma) * i / (numScales - 1.0);
            
            // Create Laplacian of Gaussian kernel
            double[][] logKernel = createLoGKernel(sigma);
            Image logResponse = Convolution.convolve(image, logKernel);
            
            // Find local maxima in scale space
            List<Point> maxima = findLocalMaxima(logResponse, threshold);
            
            for (Point point : maxima) {
                blobs.add(new Blob(point.x, point.y, sigma, point.response));
            }
        }
        
        // Remove duplicate blobs (non-maximum suppression in scale space)
        return nonMaximumSuppression3D(blobs);
    }
    
    /**
     * Create Laplacian of Gaussian kernel
     */
    private static double[][] createLoGKernel(double sigma) {
        int size = (int) Math.ceil(6 * sigma);
        if (size % 2 == 0) size++;
        
        double[][] kernel = new double[size][size];
        int center = size / 2;
        double sum = 0.0;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = x - center;
                double dy = y - center;
                double r2 = dx * dx + dy * dy;
                double sigma2 = sigma * sigma;
                
                // Laplacian of Gaussian: (r^2 - 2*sigma^2) * exp(-r^2/(2*sigma^2))
                double value = (r2 - 2 * sigma2) * Math.exp(-r2 / (2 * sigma2));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalize to sum to zero
        double mean = sum / (size * size);
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] -= mean;
            }
        }
        
        return kernel;
    }
    
    /**
     * Non-maximum suppression in 3D (x, y, scale)
     */
    private static List<Blob> nonMaximumSuppression3D(List<Blob> blobs) {
        List<Blob> result = new ArrayList<>();
        
        for (Blob blob : blobs) {
            boolean isMax = true;
            
            for (Blob other : blobs) {
                if (blob == other) continue;
                
                double distance = Math.sqrt((blob.x - other.x) * (blob.x - other.x) + 
                                          (blob.y - other.y) * (blob.y - other.y));
                double scaleDiff = Math.abs(blob.scale - other.scale);
                
                // If blobs are close in space and scale, keep the stronger one
                if (distance < 2 * blob.scale && scaleDiff < blob.scale && 
                    other.response > blob.response) {
                    isMax = false;
                    break;
                }
            }
            
            if (isMax) {
                result.add(blob);
            }
        }
        
        return result;
    }
    
    /**
     * FAST (Features from Accelerated Segment Test) corner detection
     * @param image input grayscale image
     * @param threshold intensity threshold
     * @param minContiguous minimum number of contiguous pixels
     * @return list of corner points
     */
    public static List<Point> fastCorners(Image image, int threshold, int minContiguous) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        List<Point> corners = new ArrayList<>();
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Bresenham circle for FAST-9
        int[] circleX = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
        int[] circleY = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
        
        for (int y = 3; y < height - 3; y++) {
            for (int x = 3; x < width - 3; x++) {
                double center = image.getGrayscalePixel(x, y);
                
                // Check if pixel is a corner
                if (isFastCorner(image, x, y, center, circleX, circleY, threshold, minContiguous)) {
                    corners.add(new Point(x, y, 1.0)); // FAST doesn't provide response strength
                }
            }
        }
        
        return corners;
    }
    
    /**
     * Check if a pixel is a FAST corner
     */
    private static boolean isFastCorner(Image image, int x, int y, double center, 
                                      int[] circleX, int[] circleY, int threshold, int minContiguous) {
        int brighter = 0, darker = 0;
        boolean[] brighterPixels = new boolean[16];
        boolean[] darkerPixels = new boolean[16];
        
        // Check all 16 pixels in the circle
        for (int i = 0; i < 16; i++) {
            double pixel = image.getGrayscalePixel(x + circleX[i], y + circleY[i]);
            
            if (pixel > center + threshold) {
                brighterPixels[i] = true;
                brighter++;
            } else if (pixel < center - threshold) {
                darkerPixels[i] = true;
                darker++;
            }
        }
        
        // Check for contiguous segments
        if (brighter >= minContiguous) {
            return hasContiguousSegment(brighterPixels, minContiguous);
        } else if (darker >= minContiguous) {
            return hasContiguousSegment(darkerPixels, minContiguous);
        }
        
        return false;
    }
    
    /**
     * Check if there's a contiguous segment of given length
     */
    private static boolean hasContiguousSegment(boolean[] pixels, int minLength) {
        int n = pixels.length;
        
        for (int start = 0; start < n; start++) {
            int count = 0;
            for (int i = 0; i < n; i++) {
                int idx = (start + i) % n;
                if (pixels[idx]) {
                    count++;
                    if (count >= minLength) return true;
                } else {
                    count = 0;
                }
            }
        }
        
        return false;
    }
    
    /**
     * Point class for feature locations
     */
    public static class Point {
        public final int x, y;
        public final double response;
        
        public Point(int x, int y, double response) {
            this.x = x;
            this.y = y;
            this.response = response;
        }
        
        @Override
        public String toString() {
            return String.format("Point(%d, %d, %.2f)", x, y, response);
        }
    }
    
    /**
     * Blob class for blob detection results
     */
    public static class Blob {
        public final int x, y;
        public final double scale;
        public final double response;
        
        public Blob(int x, int y, double scale, double response) {
            this.x = x;
            this.y = y;
            this.scale = scale;
            this.response = response;
        }
        
        @Override
        public String toString() {
            return String.format("Blob(%d, %d, scale=%.2f, response=%.2f)", x, y, scale, response);
        }
    }
}

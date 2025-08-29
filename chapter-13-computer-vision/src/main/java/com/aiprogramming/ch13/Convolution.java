package com.aiprogramming.ch13;

/**
 * Implements convolution operations for image filtering
 * Supports various kernel types and border handling
 */
public class Convolution {
    
    /**
     * Apply convolution kernel to an image
     * @param image input image
     * @param kernel convolution kernel
     * @return filtered image
     */
    public static Image convolve(Image image, double[][] kernel) {
        return convolve(image, kernel, BorderType.REFLECT);
    }
    
    /**
     * Apply convolution kernel to an image with specified border handling
     * @param image input image
     * @param kernel convolution kernel
     * @param borderType border handling strategy
     * @return filtered image
     */
    public static Image convolve(Image image, double[][] kernel, BorderType borderType) {
        if (image == null || kernel == null) {
            throw new IllegalArgumentException("Image and kernel cannot be null");
        }
        
        int kernelHeight = kernel.length;
        int kernelWidth = kernel[0].length;
        int kernelCenterY = kernelHeight / 2;
        int kernelCenterX = kernelWidth / 2;
        
        Image result = new Image(image.getWidth(), image.getHeight(), image.getChannels());
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double sum = 0.0;
                    
                    for (int ky = 0; ky < kernelHeight; ky++) {
                        for (int kx = 0; kx < kernelWidth; kx++) {
                            int imageY = y + ky - kernelCenterY;
                            int imageX = x + kx - kernelCenterX;
                            
                            // Handle borders
                            imageX = handleBorder(imageX, image.getWidth(), borderType);
                            imageY = handleBorder(imageY, image.getHeight(), borderType);
                            
                            if (imageX >= 0 && imageX < image.getWidth() && 
                                imageY >= 0 && imageY < image.getHeight()) {
                                sum += image.getPixel(imageX, imageY, c) * kernel[ky][kx];
                            }
                        }
                    }
                    
                    result.setPixel(x, y, c, sum);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Handle border pixels based on border type
     */
    private static int handleBorder(int coord, int size, BorderType borderType) {
        switch (borderType) {
            case ZERO:
                return coord;
            case REFLECT:
                if (coord < 0) return -coord - 1;
                if (coord >= size) return 2 * size - coord - 1;
                return coord;
            case REPLICATE:
                if (coord < 0) return 0;
                if (coord >= size) return size - 1;
                return coord;
            case WRAP:
                if (coord < 0) return size + coord;
                if (coord >= size) return coord - size;
                return coord;
            default:
                return coord;
        }
    }
    
    /**
     * Create Gaussian kernel
     * @param size kernel size (must be odd)
     * @param sigma standard deviation
     * @return Gaussian kernel
     */
    public static double[][] createGaussianKernel(int size, double sigma) {
        if (size % 2 == 0) {
            throw new IllegalArgumentException("Kernel size must be odd");
        }
        
        double[][] kernel = new double[size][size];
        int center = size / 2;
        double sum = 0.0;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = x - center;
                double dy = y - center;
                double value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalize kernel
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }
        
        return kernel;
    }
    
    /**
     * Create Sobel edge detection kernels
     * @return array containing [sobelX, sobelY] kernels
     */
    public static double[][][] createSobelKernels() {
        double[][] sobelX = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        
        double[][] sobelY = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };
        
        return new double[][][]{sobelX, sobelY};
    }
    
    /**
     * Create Laplacian kernel
     * @return Laplacian kernel
     */
    public static double[][] createLaplacianKernel() {
        return new double[][]{
            { 0, -1,  0},
            {-1,  4, -1},
            { 0, -1,  0}
        };
    }
    
    /**
     * Create mean/average filter kernel
     * @param size kernel size
     * @return mean filter kernel
     */
    public static double[][] createMeanKernel(int size) {
        double[][] kernel = new double[size][size];
        double value = 1.0 / (size * size);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] = value;
            }
        }
        
        return kernel;
    }
    
    /**
     * Apply separable convolution (more efficient for large kernels)
     * @param image input image
     * @param kernelX horizontal kernel
     * @param kernelY vertical kernel
     * @return filtered image
     */
    public static Image convolveSeparable(Image image, double[] kernelX, double[] kernelY) {
        // First apply horizontal kernel
        Image temp = convolve1D(image, kernelX, true);
        
        // Then apply vertical kernel
        return convolve1D(temp, kernelY, false);
    }
    
    /**
     * Apply 1D convolution
     * @param image input image
     * @param kernel 1D kernel
     * @param horizontal true for horizontal convolution, false for vertical
     * @return filtered image
     */
    public static Image convolve1D(Image image, double[] kernel, boolean horizontal) {
        int kernelSize = kernel.length;
        int kernelCenter = kernelSize / 2;
        
        Image result = new Image(image.getWidth(), image.getHeight(), image.getChannels());
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double sum = 0.0;
                    
                    if (horizontal) {
                        // Horizontal convolution
                        for (int k = 0; k < kernelSize; k++) {
                            int imageX = x + k - kernelCenter;
                            if (imageX >= 0 && imageX < image.getWidth()) {
                                sum += image.getPixel(imageX, y, c) * kernel[k];
                            }
                        }
                    } else {
                        // Vertical convolution
                        for (int k = 0; k < kernelSize; k++) {
                            int imageY = y + k - kernelCenter;
                            if (imageY >= 0 && imageY < image.getHeight()) {
                                sum += image.getPixel(x, imageY, c) * kernel[k];
                            }
                        }
                    }
                    
                    result.setPixel(x, y, c, sum);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Border handling strategies
     */
    public enum BorderType {
        ZERO,       // Fill with zeros
        REFLECT,    // Reflect border pixels
        REPLICATE,  // Replicate border pixels
        WRAP        // Wrap around
    }
}

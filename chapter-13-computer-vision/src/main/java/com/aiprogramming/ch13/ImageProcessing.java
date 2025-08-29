package com.aiprogramming.ch13;

import java.util.*;

/**
 * Provides various image processing operations
 * Includes filtering, morphological operations, and transformations
 */
public class ImageProcessing {
    
    /**
     * Apply Gaussian blur to an image
     * @param image input image
     * @param sigma standard deviation
     * @return blurred image
     */
    public static Image gaussianBlur(Image image, double sigma) {
        double[][] kernel = Convolution.createGaussianKernel(5, sigma);
        return Convolution.convolve(image, kernel);
    }
    
    /**
     * Apply mean filter to an image
     * @param image input image
     * @param kernelSize size of the kernel
     * @return filtered image
     */
    public static Image meanFilter(Image image, int kernelSize) {
        double[][] kernel = Convolution.createMeanKernel(kernelSize);
        return Convolution.convolve(image, kernel);
    }
    
    /**
     * Apply median filter to an image
     * @param image input image
     * @param kernelSize size of the kernel
     * @return filtered image
     */
    public static Image medianFilter(Image image, int kernelSize) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        int radius = kernelSize / 2;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                List<Double> values = new ArrayList<>();
                
                // Collect values in kernel window
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < image.getWidth() && 
                            ny >= 0 && ny < image.getHeight()) {
                            values.add(image.getGrayscalePixel(nx, ny));
                        }
                    }
                }
                
                // Compute median
                Collections.sort(values);
                double median = values.get(values.size() / 2);
                result.setGrayscalePixel(x, y, median);
            }
        }
        
        return result;
    }
    
    /**
     * Apply bilateral filter to an image
     * @param image input image
     * @param sigmaSpace spatial standard deviation
     * @param sigmaColor color standard deviation
     * @return filtered image
     */
    public static Image bilateralFilter(Image image, double sigmaSpace, double sigmaColor) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        int radius = (int) Math.ceil(3 * sigmaSpace);
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double centerValue = image.getGrayscalePixel(x, y);
                double sum = 0.0;
                double weightSum = 0.0;
                
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < image.getWidth() && 
                            ny >= 0 && ny < image.getHeight()) {
                            
                            double neighborValue = image.getGrayscalePixel(nx, ny);
                            
                            // Spatial weight
                            double spatialDist = Math.sqrt(kx * kx + ky * ky);
                            double spatialWeight = Math.exp(-spatialDist * spatialDist / 
                                                          (2 * sigmaSpace * sigmaSpace));
                            
                            // Color weight
                            double colorDist = Math.abs(centerValue - neighborValue);
                            double colorWeight = Math.exp(-colorDist * colorDist / 
                                                        (2 * sigmaColor * sigmaColor));
                            
                            double weight = spatialWeight * colorWeight;
                            sum += weight * neighborValue;
                            weightSum += weight;
                        }
                    }
                }
                
                result.setGrayscalePixel(x, y, sum / weightSum);
            }
        }
        
        return result;
    }
    
    /**
     * Apply morphological erosion
     * @param image input binary image
     * @param kernelSize size of structuring element
     * @return eroded image
     */
    public static Image erode(Image image, int kernelSize) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        int radius = kernelSize / 2;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double minValue = Double.MAX_VALUE;
                
                // Find minimum value in kernel window
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < image.getWidth() && 
                            ny >= 0 && ny < image.getHeight()) {
                            double value = image.getGrayscalePixel(nx, ny);
                            minValue = Math.min(minValue, value);
                        }
                    }
                }
                
                result.setGrayscalePixel(x, y, minValue);
            }
        }
        
        return result;
    }
    
    /**
     * Apply morphological dilation
     * @param image input binary image
     * @param kernelSize size of structuring element
     * @return dilated image
     */
    public static Image dilate(Image image, int kernelSize) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        int radius = kernelSize / 2;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double maxValue = Double.MIN_VALUE;
                
                // Find maximum value in kernel window
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < image.getWidth() && 
                            ny >= 0 && ny < image.getHeight()) {
                            double value = image.getGrayscalePixel(nx, ny);
                            maxValue = Math.max(maxValue, value);
                        }
                    }
                }
                
                result.setGrayscalePixel(x, y, maxValue);
            }
        }
        
        return result;
    }
    
    /**
     * Apply morphological opening (erosion followed by dilation)
     * @param image input image
     * @param kernelSize size of structuring element
     * @return opened image
     */
    public static Image open(Image image, int kernelSize) {
        Image eroded = erode(image, kernelSize);
        return dilate(eroded, kernelSize);
    }
    
    /**
     * Apply morphological closing (dilation followed by erosion)
     * @param image input image
     * @param kernelSize size of structuring element
     * @return closed image
     */
    public static Image close(Image image, int kernelSize) {
        Image dilated = dilate(image, kernelSize);
        return erode(dilated, kernelSize);
    }
    
    /**
     * Apply thresholding to create binary image
     * @param image input grayscale image
     * @param threshold threshold value
     * @return binary image
     */
    public static Image threshold(Image image, double threshold) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double value = image.getGrayscalePixel(x, y);
                double binaryValue = (value > threshold) ? 255.0 : 0.0;
                result.setGrayscalePixel(x, y, binaryValue);
            }
        }
        
        return result;
    }
    
    /**
     * Apply adaptive thresholding using local mean
     * @param image input grayscale image
     * @param blockSize size of local neighborhood
     * @param constant constant subtracted from mean
     * @return binary image
     */
    public static Image adaptiveThreshold(Image image, int blockSize, double constant) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        int radius = blockSize / 2;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double sum = 0.0;
                int count = 0;
                
                // Compute local mean
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < image.getWidth() && 
                            ny >= 0 && ny < image.getHeight()) {
                            sum += image.getGrayscalePixel(nx, ny);
                            count++;
                        }
                    }
                }
                
                double localMean = sum / count;
                double threshold = localMean - constant;
                double value = image.getGrayscalePixel(x, y);
                double binaryValue = (value > threshold) ? 255.0 : 0.0;
                result.setGrayscalePixel(x, y, binaryValue);
            }
        }
        
        return result;
    }
    
    /**
     * Apply histogram equalization
     * @param image input grayscale image
     * @return equalized image
     */
    public static Image histogramEqualization(Image image) {
        if (!image.isGrayscale()) {
            image = image.toGrayscale();
        }
        
        // Compute histogram
        int[] histogram = new int[256];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int value = (int) image.getGrayscalePixel(x, y);
                histogram[value]++;
            }
        }
        
        // Compute cumulative distribution function
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }
        
        // Normalize CDF
        int totalPixels = image.getWidth() * image.getHeight();
        double[] normalizedCdf = new double[256];
        for (int i = 0; i < 256; i++) {
            normalizedCdf[i] = (double) cdf[i] / totalPixels;
        }
        
        // Apply equalization
        Image result = new Image(image.getWidth(), image.getHeight(), 1);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int value = (int) image.getGrayscalePixel(x, y);
                double equalizedValue = normalizedCdf[value] * 255.0;
                result.setGrayscalePixel(x, y, equalizedValue);
            }
        }
        
        return result;
    }
    
    /**
     * Apply gamma correction
     * @param image input image
     * @param gamma gamma value
     * @return corrected image
     */
    public static Image gammaCorrection(Image image, double gamma) {
        Image result = new Image(image.getWidth(), image.getHeight(), image.getChannels());
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double value = image.getPixel(x, y, c);
                    double corrected = Math.pow(value / 255.0, gamma) * 255.0;
                    result.setPixel(x, y, c, corrected);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Apply contrast stretching
     * @param image input image
     * @param minOutput minimum output value
     * @param maxOutput maximum output value
     * @return stretched image
     */
    public static Image contrastStretching(Image image, double minOutput, double maxOutput) {
        // Find min and max values
        double minInput = Double.MAX_VALUE;
        double maxInput = Double.MIN_VALUE;
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double value = image.getPixel(x, y, c);
                    minInput = Math.min(minInput, value);
                    maxInput = Math.max(maxInput, value);
                }
            }
        }
        
        // Apply linear stretching
        Image result = new Image(image.getWidth(), image.getHeight(), image.getChannels());
        double scale = (maxOutput - minOutput) / (maxInput - minInput);
        
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                for (int c = 0; c < image.getChannels(); c++) {
                    double value = image.getPixel(x, y, c);
                    double stretched = minOutput + scale * (value - minInput);
                    result.setPixel(x, y, c, stretched);
                }
            }
        }
        
        return result;
    }
}

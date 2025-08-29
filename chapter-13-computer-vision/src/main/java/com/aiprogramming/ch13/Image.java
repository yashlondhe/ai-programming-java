package com.aiprogramming.ch13;

import java.util.Arrays;

/**
 * Represents a digital image with pixel data
 * Supports grayscale and RGB color images
 */
public class Image {
    
    private final int width;
    private final int height;
    private final int channels;
    private final double[][][] pixels;
    
    /**
     * Create a new image with specified dimensions
     * @param width image width in pixels
     * @param height image height in pixels
     * @param channels number of color channels (1 for grayscale, 3 for RGB)
     */
    public Image(int width, int height, int channels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.pixels = new double[height][width][channels];
    }
    
    /**
     * Create a grayscale image
     */
    public static Image createGrayscale(int width, int height) {
        return new Image(width, height, 1);
    }
    
    /**
     * Create an RGB image
     */
    public static Image createRGB(int width, int height) {
        return new Image(width, height, 3);
    }
    
    /**
     * Get pixel value at specified position and channel
     */
    public double getPixel(int x, int y, int channel) {
        if (x < 0 || x >= width || y < 0 || y >= height || channel < 0 || channel >= channels) {
            throw new IllegalArgumentException("Invalid coordinates or channel");
        }
        return pixels[y][x][channel];
    }
    
    /**
     * Set pixel value at specified position and channel
     */
    public void setPixel(int x, int y, int channel, double value) {
        if (x < 0 || x >= width || y < 0 || y >= height || channel < 0 || channel >= channels) {
            throw new IllegalArgumentException("Invalid coordinates or channel");
        }
        pixels[y][x][channel] = Math.max(0.0, Math.min(255.0, value));
    }
    
    /**
     * Get grayscale pixel value (for grayscale images)
     */
    public double getGrayscalePixel(int x, int y) {
        if (channels != 1) {
            throw new IllegalStateException("Image is not grayscale");
        }
        return getPixel(x, y, 0);
    }
    
    /**
     * Set grayscale pixel value (for grayscale images)
     */
    public void setGrayscalePixel(int x, int y, double value) {
        if (channels != 1) {
            throw new IllegalStateException("Image is not grayscale");
        }
        setPixel(x, y, 0, value);
    }
    
    /**
     * Get RGB pixel values
     */
    public double[] getRGBPixel(int x, int y) {
        if (channels != 3) {
            throw new IllegalStateException("Image is not RGB");
        }
        return new double[]{
            getPixel(x, y, 0), // Red
            getPixel(x, y, 1), // Green
            getPixel(x, y, 2)  // Blue
        };
    }
    
    /**
     * Set RGB pixel values
     */
    public void setRGBPixel(int x, int y, double r, double g, double b) {
        if (channels != 3) {
            throw new IllegalStateException("Image is not RGB");
        }
        setPixel(x, y, 0, r);
        setPixel(x, y, 1, g);
        setPixel(x, y, 2, b);
    }
    
    /**
     * Convert RGB image to grayscale using luminance formula
     */
    public Image toGrayscale() {
        if (channels != 3) {
            throw new IllegalStateException("Image is not RGB");
        }
        
        Image grayscale = Image.createGrayscale(width, height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double[] rgb = getRGBPixel(x, y);
                // Luminance formula: 0.299*R + 0.587*G + 0.114*B
                double gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                grayscale.setGrayscalePixel(x, y, gray);
            }
        }
        
        return grayscale;
    }
    
    /**
     * Create a copy of this image
     */
    public Image copy() {
        Image copy = new Image(width, height, channels);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    copy.setPixel(x, y, c, getPixel(x, y, c));
                }
            }
        }
        return copy;
    }
    
    /**
     * Fill the entire image with a constant value
     */
    public void fill(double value) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    setPixel(x, y, c, value);
                }
            }
        }
    }
    
    /**
     * Normalize pixel values to range [0, 255]
     */
    public void normalize() {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        // Find min and max values
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    double value = getPixel(x, y, c);
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }
        }
        
        // Normalize to [0, 255]
        if (max > min) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        double value = getPixel(x, y, c);
                        double normalized = 255.0 * (value - min) / (max - min);
                        setPixel(x, y, c, normalized);
                    }
                }
            }
        }
    }
    
    // Getters
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getChannels() { return channels; }
    public boolean isGrayscale() { return channels == 1; }
    public boolean isRGB() { return channels == 3; }
    
    @Override
    public String toString() {
        return String.format("Image[%dx%d, %d channels]", width, height, channels);
    }
}

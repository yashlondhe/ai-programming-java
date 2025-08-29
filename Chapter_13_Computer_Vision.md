# Chapter 13: Computer Vision

## Introduction

Computer Vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves developing algorithms and systems that can process, analyze, and extract meaningful information from images and videos. Computer vision has applications in autonomous vehicles, medical imaging, robotics, security systems, and many other domains.

### Learning Objectives

By the end of this chapter, you will be able to:

- Understand fundamental computer vision concepts and image representation
- Implement various convolution operations and filtering techniques
- Apply multiple edge detection algorithms (Sobel, Canny, Laplacian, etc.)
- Detect features and corners using Harris, FAST, and blob detection methods
- Perform morphological operations and image enhancement
- Build complete image processing pipelines
- Compare the performance of different computer vision algorithms

### Key Concepts

- **Image Representation**: Digital images as 2D arrays of pixel values
- **Convolution**: Mathematical operation for filtering and feature extraction
- **Edge Detection**: Identifying boundaries and transitions in images
- **Feature Detection**: Finding distinctive points and regions in images
- **Morphological Operations**: Shape-based image processing techniques
- **Image Enhancement**: Improving image quality and visibility

## 13.1 Image Representation

Digital images are represented as 2D arrays of pixel values, where each pixel contains intensity or color information.

### 13.1.1 Image Class Implementation

```java
package com.aiprogramming.ch13;

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
    
    // Additional methods for image manipulation...
}
```

### 13.1.2 Color Space Conversion

The RGB to grayscale conversion uses the luminance formula that accounts for human visual sensitivity to different colors:

```
Gray = 0.299 × Red + 0.587 × Green + 0.114 × Blue
```

This formula is based on the fact that the human eye is most sensitive to green light, followed by red, and least sensitive to blue.

## 13.2 Convolution Operations

Convolution is a fundamental operation in computer vision that applies a kernel (filter) to an image to extract features or modify the image.

### 13.2.1 Convolution Implementation

```java
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
}
```

### 13.2.2 Common Convolution Kernels

#### Gaussian Kernel

```java
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
```

#### Sobel Kernels

```java
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
```

## 13.3 Edge Detection

Edge detection is a fundamental computer vision technique that identifies boundaries and transitions in images.

### 13.3.1 Sobel Edge Detection

Sobel edge detection uses gradient operators to detect edges by computing the magnitude of the gradient.

```java
package com.aiprogramming.ch13;

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
}
```

#### Key Features

- **Gradient Computation**: Uses Sobel operators to compute x and y gradients
- **Magnitude Calculation**: Computes gradient magnitude using Euclidean distance
- **Direction Information**: Can also compute gradient direction for advanced analysis
- **Noise Sensitivity**: Moderately sensitive to noise, good for most applications

### 13.3.2 Canny Edge Detection

Canny edge detection is a multi-stage algorithm that produces high-quality edges with noise suppression.

```java
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
```

#### Algorithm Steps

1. **Gaussian Smoothing**: Reduce noise while preserving edges
2. **Gradient Computation**: Calculate gradient magnitude and direction
3. **Non-maximum Suppression**: Thin edges by keeping only local maxima
4. **Hysteresis Thresholding**: Connect strong edges and remove weak ones

### 13.3.3 Laplacian Edge Detection

Laplacian edge detection uses second-order derivatives to detect edges.

```java
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
```

#### Characteristics

- **Second-order Derivative**: Detects zero-crossings in the second derivative
- **Noise Sensitivity**: Very sensitive to noise, requires preprocessing
- **Fine Detail**: Good at detecting fine edges and details
- **Direction Invariant**: Responds to edges in all directions

## 13.4 Feature Detection

Feature detection identifies distinctive points and regions in images that can be used for matching and recognition.

### 13.4.1 Harris Corner Detection

Harris corner detection identifies corners by analyzing the structure tensor of the image.

```java
package com.aiprogramming.ch13;

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
}
```

#### Harris Corner Response

The Harris corner response is computed as:

```
R = det(M) - k × trace(M)²
```

where M is the structure tensor:

```
M = [Ixx  Ixy]
    [Ixy  Iyy]
```

### 13.4.2 FAST Corner Detection

FAST (Features from Accelerated Segment Test) is a high-speed corner detection algorithm.

```java
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
                corners.add(new Point(x, y, 1.0));
            }
        }
    }
    
    return corners;
}
```

#### FAST Algorithm

1. **Circle Test**: Compare center pixel with 16 pixels in a circle
2. **Contiguous Segment**: Look for contiguous segments of brighter/darker pixels
3. **Speed Optimization**: Early termination for non-corners
4. **High Performance**: Much faster than Harris corner detection

### 13.4.3 Blob Detection

Blob detection identifies regions of interest using scale-space analysis.

```java
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
```

## 13.5 Image Processing Operations

Image processing operations enhance, filter, and transform images for better analysis.

### 13.5.1 Spatial Filtering

```java
package com.aiprogramming.ch13;

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
}
```

### 13.5.2 Morphological Operations

Morphological operations process images based on shape and structure.

```java
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
```

### 13.5.3 Image Enhancement

```java
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
```

## 13.6 Practical Example

### 13.6.1 Comprehensive Demonstration

```java
package com.aiprogramming.ch13;

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
    
    // Additional demonstration methods...
}
```

### 13.6.2 Expected Results

The example typically produces:

1. **Image Processing**: Various filtering operations with different effects
2. **Edge Detection**: Multiple edge maps showing different characteristics
3. **Feature Detection**: Corner and blob locations with response values
4. **Advanced Operations**: Morphological and enhancement results

## 13.7 Advanced Topics

### 13.7.1 Scale-Space Analysis

- **Gaussian Pyramid**: Multi-scale image representation
- **Laplacian Pyramid**: Difference of Gaussians for feature detection
- **Scale-Invariant Features**: Features that are robust to scale changes

### 13.7.2 Feature Descriptors

- **SIFT**: Scale-Invariant Feature Transform
- **SURF**: Speeded Up Robust Features
- **ORB**: Oriented FAST and Rotated BRIEF

### 13.7.3 Deep Learning Integration

- **Convolutional Neural Networks**: Deep learning for computer vision
- **Transfer Learning**: Using pre-trained models
- **Object Detection**: YOLO, R-CNN, and similar architectures

## 13.8 Best Practices

### 13.8.1 Algorithm Selection

- **Edge Detection**: Choose based on noise level and edge characteristics
- **Feature Detection**: Select based on application requirements
- **Filtering**: Consider noise type and preservation requirements

### 13.8.2 Parameter Tuning

- **Threshold Values**: Start with standard values and adjust based on results
- **Kernel Sizes**: Balance between effectiveness and computational cost
- **Multiple Scales**: Use scale-space approaches for robust detection

### 13.8.3 Performance Optimization

- **Separable Convolution**: Use for large kernels to improve efficiency
- **Memory Management**: Consider memory usage for large images
- **Parallel Processing**: Utilize multi-threading for independent operations

## 13.9 Summary

In this chapter, we explored fundamental computer vision concepts:

1. **Image Representation**: Digital images as pixel arrays with color space conversion
2. **Convolution Operations**: Core filtering with various kernels and border handling
3. **Edge Detection**: Multiple algorithms for boundary detection and analysis
4. **Feature Detection**: Corner and blob detection for interest point identification
5. **Image Processing**: Filtering, morphological operations, and enhancement techniques
6. **Practical Applications**: Complete computer vision pipelines and demonstrations

### Key Takeaways

- **Computer Vision** enables machines to interpret and understand visual information
- **Convolution** is the fundamental operation for image filtering and feature extraction
- **Edge Detection** provides boundary information essential for object recognition
- **Feature Detection** identifies distinctive points for matching and tracking
- **Image Processing** enhances and transforms images for better analysis

### Next Steps

- Explore deep learning-based computer vision
- Implement object detection and recognition algorithms
- Study image segmentation and instance segmentation
- Learn about optical flow and motion detection
- Investigate 3D computer vision and stereo vision
- Apply computer vision to real-world applications

## Exercises

### Exercise 1: Custom Kernel Design
Create custom convolution kernels for specific applications (sharpening, embossing, etc.).

### Exercise 2: Edge Detection Comparison
Compare the performance of different edge detection algorithms on various image types.

### Exercise 3: Feature Detection Analysis
Analyze the performance of Harris, FAST, and blob detection on different images.

### Exercise 4: Image Processing Pipeline
Build a complete image processing pipeline for a specific application.

### Exercise 5: Performance Optimization
Optimize the convolution operations for better performance on large images.

### Exercise 6: Real-World Application
Apply the computer vision algorithms to a practical problem (e.g., document analysis, medical imaging).

### Exercise 7: Advanced Feature Descriptors
Implement SIFT or SURF feature descriptors for robust feature matching.

### Exercise 8: Deep Learning Integration
Integrate the traditional computer vision algorithms with deep learning approaches.

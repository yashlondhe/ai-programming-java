package com.aiprogramming.ch08;

/**
 * Pooling operations for CNNs
 */
public class Pooling {
    
    /**
     * Max pooling operation
     */
    public static Tensor maxPool(Tensor input, int poolSize, int stride) {
        int[] inputShape = input.getShape();
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Input must be a 3D tensor (channels, height, width)");
        }
        
        int channels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        
        // Calculate output dimensions
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        
        Tensor output = Tensor.zeros(channels, outputHeight, outputWidth);
        Tensor mask = Tensor.zeros(inputShape); // For backpropagation
        
        for (int c = 0; c < channels; c++) {
            for (int outH = 0; outH < outputHeight; outH++) {
                for (int outW = 0; outW < outputWidth; outW++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxH = -1, maxW = -1;
                    
                    // Find maximum in pooling window
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int inH = outH * stride + ph;
                            int inW = outW * stride + pw;
                            
                            if (inH < inputHeight && inW < inputWidth) {
                                double val = input.get(c, inH, inW);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = inH;
                                    maxW = inW;
                                }
                            }
                        }
                    }
                    
                    output.set(maxVal, c, outH, outW);
                    
                    // Store mask for backpropagation
                    if (maxH >= 0 && maxW >= 0) {
                        mask.set(1.0, c, maxH, maxW);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * Average pooling operation
     */
    public static Tensor averagePool(Tensor input, int poolSize, int stride) {
        int[] inputShape = input.getShape();
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Input must be a 3D tensor (channels, height, width)");
        }
        
        int channels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        
        // Calculate output dimensions
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        
        Tensor output = Tensor.zeros(channels, outputHeight, outputWidth);
        
        for (int c = 0; c < channels; c++) {
            for (int outH = 0; outH < outputHeight; outH++) {
                for (int outW = 0; outW < outputWidth; outW++) {
                    double sum = 0.0;
                    int count = 0;
                    
                    // Calculate average in pooling window
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int inH = outH * stride + ph;
                            int inW = outW * stride + pw;
                            
                            if (inH < inputHeight && inW < inputWidth) {
                                sum += input.get(c, inH, inW);
                                count++;
                            }
                        }
                    }
                    
                    double avgVal = count > 0 ? sum / count : 0.0;
                    output.set(avgVal, c, outH, outW);
                }
            }
        }
        
        return output;
    }
    
    /**
     * Global average pooling - reduces spatial dimensions to 1x1
     */
    public static Tensor globalAveragePool(Tensor input) {
        int[] inputShape = input.getShape();
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Input must be a 3D tensor (channels, height, width)");
        }
        
        int channels = inputShape[0];
        int height = inputShape[1];
        int width = inputShape[2];
        
        Tensor output = Tensor.zeros(channels, 1, 1);
        
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            int totalPixels = height * width;
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    sum += input.get(c, h, w);
                }
            }
            
            double avgVal = sum / totalPixels;
            output.set(avgVal, c, 0, 0);
        }
        
        return output;
    }
    
    /**
     * Global max pooling - reduces spatial dimensions to 1x1
     */
    public static Tensor globalMaxPool(Tensor input) {
        int[] inputShape = input.getShape();
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Input must be a 3D tensor (channels, height, width)");
        }
        
        int channels = inputShape[0];
        int height = inputShape[1];
        int width = inputShape[2];
        
        Tensor output = Tensor.zeros(channels, 1, 1);
        
        for (int c = 0; c < channels; c++) {
            double maxVal = Double.NEGATIVE_INFINITY;
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double val = input.get(c, h, w);
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
            
            output.set(maxVal, c, 0, 0);
        }
        
        return output;
    }
    
    /**
     * Compute gradients for max pooling backpropagation
     */
    public static Tensor maxPoolGradient(Tensor input, Tensor outputGradient, int poolSize, int stride) {
        int[] inputShape = input.getShape();
        int[] outputShape = outputGradient.getShape();
        
        int channels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];
        
        Tensor inputGradient = Tensor.zeros(inputShape);
        
        for (int c = 0; c < channels; c++) {
            for (int outH = 0; outH < outputHeight; outH++) {
                for (int outW = 0; outW < outputWidth; outW++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxH = -1, maxW = -1;
                    
                    // Find maximum in pooling window
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int inH = outH * stride + ph;
                            int inW = outW * stride + pw;
                            
                            if (inH < inputHeight && inW < inputWidth) {
                                double val = input.get(c, inH, inW);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = inH;
                                    maxW = inW;
                                }
                            }
                        }
                    }
                    
                    // Propagate gradient to the maximum position
                    if (maxH >= 0 && maxW >= 0) {
                        double currentGrad = inputGradient.get(c, maxH, maxW);
                        double outputGrad = outputGradient.get(c, outH, outW);
                        inputGradient.set(currentGrad + outputGrad, c, maxH, maxW);
                    }
                }
            }
        }
        
        return inputGradient;
    }
    
    /**
     * Compute gradients for average pooling backpropagation
     */
    public static Tensor averagePoolGradient(Tensor input, Tensor outputGradient, int poolSize, int stride) {
        int[] inputShape = input.getShape();
        int[] outputShape = outputGradient.getShape();
        
        int channels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];
        
        Tensor inputGradient = Tensor.zeros(inputShape);
        
        for (int c = 0; c < channels; c++) {
            for (int outH = 0; outH < outputHeight; outH++) {
                for (int outW = 0; outW < outputWidth; outW++) {
                    int count = 0;
                    
                    // Count valid positions in pooling window
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int inH = outH * stride + ph;
                            int inW = outW * stride + pw;
                            
                            if (inH < inputHeight && inW < inputWidth) {
                                count++;
                            }
                        }
                    }
                    
                    // Distribute gradient equally to all positions
                    double gradientPerPosition = count > 0 ? outputGradient.get(c, outH, outW) / count : 0.0;
                    
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int inH = outH * stride + ph;
                            int inW = outW * stride + pw;
                            
                            if (inH < inputHeight && inW < inputWidth) {
                                double currentGrad = inputGradient.get(c, inH, inW);
                                inputGradient.set(currentGrad + gradientPerPosition, c, inH, inW);
                            }
                        }
                    }
                }
            }
        }
        
        return inputGradient;
    }
}

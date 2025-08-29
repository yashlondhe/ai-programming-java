package com.aiprogramming.ch08;

import java.util.Random;

/**
 * Abstract base class for CNN layers
 */
public abstract class CNNLayer {
    protected Tensor lastInput;
    protected Tensor lastOutput;
    
    public abstract Tensor forward(Tensor input);
    public abstract Tensor backward(Tensor outputGradient);
    public abstract void updateWeights(double learningRate);
    
    public Tensor getLastOutput() {
        return lastOutput;
    }
    
    public Tensor getLastInput() {
        return lastInput;
    }
}

/**
 * Convolutional layer implementation
 */
class Conv2DLayer extends CNNLayer {
    private Tensor[] kernels; // [numFilters, channels, height, width]
    private Tensor[] biases;
    private Tensor[] kernelGradients;
    private Tensor[] biasGradients;
    private int stride;
    private String padding;
    private int numFilters;
    private int kernelSize;
    private int inputChannels;
    
    public Conv2DLayer(int numFilters, int kernelSize, int inputChannels, int stride, String padding) {
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.inputChannels = inputChannels;
        this.stride = stride;
        this.padding = padding;
        
        // Initialize kernels and biases
        this.kernels = new Tensor[numFilters];
        this.biases = new Tensor[numFilters];
        this.kernelGradients = new Tensor[numFilters];
        this.biasGradients = new Tensor[numFilters];
        
        Random random = new Random();
        for (int i = 0; i < numFilters; i++) {
            // Initialize kernels with small random values
            kernels[i] = Tensor.random(inputChannels, kernelSize, kernelSize);
            for (int j = 0; j < kernels[i].getTotalSize(); j++) {
                kernels[i].getData()[j] *= 0.1; // Scale down initial weights
            }
            
            // Initialize biases to zero
            biases[i] = Tensor.zeros(1, 1, 1);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        
        // Apply convolution with each kernel
        Tensor[] outputs = new Tensor[numFilters];
        for (int i = 0; i < numFilters; i++) {
            outputs[i] = Convolution.convolve2D(input, kernels[i], stride, padding);
            
            // Add bias
            int[] outputShape = outputs[i].getShape();
            for (int h = 0; h < outputShape[1]; h++) {
                for (int w = 0; w < outputShape[2]; w++) {
                    double currentVal = outputs[i].get(0, h, w);
                    double biasVal = biases[i].get(0, 0, 0);
                    outputs[i].set(currentVal + biasVal, 0, h, w);
                }
            }
        }
        
        // Stack outputs along channel dimension
        this.lastOutput = stackTensors(outputs, 0);
        return this.lastOutput;
    }
    
    @Override
    public Tensor backward(Tensor outputGradient) {
        int[] outputShape = outputGradient.getShape();
        int numChannels = outputShape[0];
        
        // Initialize gradients
        for (int i = 0; i < numFilters; i++) {
            kernelGradients[i] = Tensor.zeros(kernels[i].getShape());
            biasGradients[i] = Tensor.zeros(biases[i].getShape());
        }
        
        // Compute gradients for each filter
        Tensor inputGradient = Tensor.zeros(lastInput.getShape());
        
        for (int i = 0; i < numFilters; i++) {
            // Extract gradient for this filter
            Tensor filterGradient = extractChannel(outputGradient, i);
            
            // Compute convolution gradients
            Convolution.ConvolutionGradients convGrads = 
                Convolution.computeGradients(lastInput, kernels[i], filterGradient, stride, padding);
            
            // Accumulate input gradients
            inputGradient = inputGradient.add(convGrads.inputGradient);
            
            // Store kernel gradients
            kernelGradients[i] = convGrads.kernelGradient;
            
            // Compute bias gradients
            double biasGrad = 0.0;
            for (int h = 0; h < filterGradient.getShape()[1]; h++) {
                for (int w = 0; w < filterGradient.getShape()[2]; w++) {
                    biasGrad += filterGradient.get(0, h, w);
                }
            }
            biasGradients[i].set(biasGrad, 0, 0, 0);
        }
        
        return inputGradient;
    }
    
    @Override
    public void updateWeights(double learningRate) {
        for (int i = 0; i < numFilters; i++) {
            // Update kernels
            double[] kernelData = kernels[i].getData();
            double[] kernelGradData = kernelGradients[i].getData();
            for (int j = 0; j < kernelData.length; j++) {
                kernelData[j] -= learningRate * kernelGradData[j];
            }
            
            // Update biases
            double biasData = biases[i].get(0, 0, 0);
            double biasGradData = biasGradients[i].get(0, 0, 0);
            biases[i].set(biasData - learningRate * biasGradData, 0, 0, 0);
        }
    }
    
    private Tensor stackTensors(Tensor[] tensors, int dimension) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("Cannot stack empty array of tensors");
        }
        
        int[] firstShape = tensors[0].getShape();
        for (Tensor tensor : tensors) {
            if (!java.util.Arrays.equals(tensor.getShape(), firstShape)) {
                throw new IllegalArgumentException("All tensors must have the same shape");
            }
        }
        
        int[] newShape = new int[firstShape.length + 1];
        for (int i = 0; i < dimension; i++) {
            newShape[i] = firstShape[i];
        }
        newShape[dimension] = tensors.length;
        for (int i = dimension; i < firstShape.length; i++) {
            newShape[i + 1] = firstShape[i];
        }
        
        Tensor result = new Tensor(newShape);
        int tensorSize = tensors[0].getTotalSize();
        
        for (int i = 0; i < tensors.length; i++) {
            double[] tensorData = tensors[i].getData();
            for (int j = 0; j < tensorSize; j++) {
                int resultIndex = i * tensorSize + j;
                result.getData()[resultIndex] = tensorData[j];
            }
        }
        
        return result;
    }
    
    private Tensor extractChannel(Tensor tensor, int channel) {
        int[] shape = tensor.getShape();
        int height = shape[1];
        int width = shape[2];
        
        Tensor result = Tensor.zeros(1, height, width);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                double val = tensor.get(channel, h, w);
                result.set(val, 0, h, w);
            }
        }
        
        return result;
    }
}

/**
 * Max pooling layer implementation
 */
class MaxPoolingLayer extends CNNLayer {
    private int poolSize;
    private int stride;
    
    public MaxPoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        this.lastOutput = Pooling.maxPool(input, poolSize, stride);
        return this.lastOutput;
    }
    
    @Override
    public Tensor backward(Tensor outputGradient) {
        return Pooling.maxPoolGradient(lastInput, outputGradient, poolSize, stride);
    }
    
    @Override
    public void updateWeights(double learningRate) {
        // Max pooling has no trainable parameters
    }
}

/**
 * Average pooling layer implementation
 */
class AveragePoolingLayer extends CNNLayer {
    private int poolSize;
    private int stride;
    
    public AveragePoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        this.lastOutput = Pooling.averagePool(input, poolSize, stride);
        return this.lastOutput;
    }
    
    @Override
    public Tensor backward(Tensor outputGradient) {
        return Pooling.averagePoolGradient(lastInput, outputGradient, poolSize, stride);
    }
    
    @Override
    public void updateWeights(double learningRate) {
        // Average pooling has no trainable parameters
    }
}

/**
 * Global average pooling layer implementation
 */
class GlobalAveragePoolingLayer extends CNNLayer {
    
    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        this.lastOutput = Pooling.globalAveragePool(input);
        return this.lastOutput;
    }
    
    @Override
    public Tensor backward(Tensor outputGradient) {
        int[] inputShape = lastInput.getShape();
        int[] outputShape = outputGradient.getShape();
        
        int channels = inputShape[0];
        int height = inputShape[1];
        int width = inputShape[2];
        
        Tensor inputGradient = Tensor.zeros(inputShape);
        
        for (int c = 0; c < channels; c++) {
            double outputGrad = outputGradient.get(c, 0, 0);
            double gradientPerPixel = outputGrad / (height * width);
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    inputGradient.set(gradientPerPixel, c, h, w);
                }
            }
        }
        
        return inputGradient;
    }
    
    @Override
    public void updateWeights(double learningRate) {
        // Global average pooling has no trainable parameters
    }
}

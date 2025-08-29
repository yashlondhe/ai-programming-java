package com.aiprogramming.ch08;

/**
 * 2D Convolution operation for CNNs
 */
public class Convolution {
    
    /**
     * Perform 2D convolution with padding
     */
    public static Tensor convolve2D(Tensor input, Tensor kernel, int stride, String padding) {
        int[] inputShape = input.getShape();
        int[] kernelShape = kernel.getShape();
        
        if (inputShape.length != 3 || kernelShape.length != 3) {
            throw new IllegalArgumentException("Input and kernel must be 3D tensors (channels, height, width)");
        }
        
        int inputChannels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        
        int kernelChannels = kernelShape[0];
        int kernelHeight = kernelShape[1];
        int kernelWidth = kernelShape[2];
        
        if (inputChannels != kernelChannels) {
            throw new IllegalArgumentException("Input and kernel must have same number of channels");
        }
        
        // Calculate output dimensions
        int outputHeight, outputWidth;
        int padHeight, padWidth;
        
        if ("same".equals(padding)) {
            outputHeight = inputHeight;
            outputWidth = inputWidth;
            padHeight = (kernelHeight - 1) / 2;
            padWidth = (kernelWidth - 1) / 2;
        } else if ("valid".equals(padding)) {
            outputHeight = (inputHeight - kernelHeight) / stride + 1;
            outputWidth = (inputWidth - kernelWidth) / stride + 1;
            padHeight = 0;
            padWidth = 0;
        } else {
            throw new IllegalArgumentException("Padding must be 'same' or 'valid'");
        }
        
        // Create output tensor
        Tensor output = Tensor.zeros(1, outputHeight, outputWidth);
        
        // Perform convolution
        for (int outH = 0; outH < outputHeight; outH++) {
            for (int outW = 0; outW < outputWidth; outW++) {
                double sum = 0.0;
                
                for (int c = 0; c < inputChannels; c++) {
                    for (int kh = 0; kh < kernelHeight; kh++) {
                        for (int kw = 0; kw < kernelWidth; kw++) {
                            int inH = outH * stride + kh - padHeight;
                            int inW = outW * stride + kw - padWidth;
                            
                            // Check bounds
                            if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                                double inputVal = input.get(c, inH, inW);
                                double kernelVal = kernel.get(c, kh, kw);
                                sum += inputVal * kernelVal;
                            }
                        }
                    }
                }
                
                output.set(sum, 0, outH, outW);
            }
        }
        
        return output;
    }
    
    /**
     * Perform 2D convolution with multiple kernels (filters)
     */
    public static Tensor convolve2D(Tensor input, Tensor[] kernels, int stride, String padding) {
        int numKernels = kernels.length;
        Tensor[] outputs = new Tensor[numKernels];
        
        for (int i = 0; i < numKernels; i++) {
            outputs[i] = convolve2D(input, kernels[i], stride, padding);
        }
        
        // Stack outputs along channel dimension
        return stackTensors(outputs, 0);
    }
    
    /**
     * Compute gradients for backpropagation
     */
    public static ConvolutionGradients computeGradients(Tensor input, Tensor kernel, 
                                                       Tensor outputGradient, int stride, String padding) {
        int[] inputShape = input.getShape();
        int[] kernelShape = kernel.getShape();
        int[] outputShape = outputGradient.getShape();
        
        int inputChannels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        
        int kernelHeight = kernelShape[1];
        int kernelWidth = kernelShape[2];
        
        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];
        
        // Calculate padding
        int padHeight, padWidth;
        if ("same".equals(padding)) {
            padHeight = (kernelHeight - 1) / 2;
            padWidth = (kernelWidth - 1) / 2;
        } else {
            padHeight = 0;
            padWidth = 0;
        }
        
        // Initialize gradients
        Tensor inputGradient = Tensor.zeros(inputShape);
        Tensor kernelGradient = Tensor.zeros(kernelShape);
        
        // Compute input gradients
        for (int c = 0; c < inputChannels; c++) {
            for (int inH = 0; inH < inputHeight; inH++) {
                for (int inW = 0; inW < inputWidth; inW++) {
                    double gradSum = 0.0;
                    
                    for (int outH = 0; outH < outputHeight; outH++) {
                        for (int outW = 0; outW < outputWidth; outW++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int expectedInH = outH * stride + kh - padHeight;
                                    int expectedInW = outW * stride + kw - padWidth;
                                    
                                    if (expectedInH == inH && expectedInW == inW) {
                                        double kernelVal = kernel.get(c, kh, kw);
                                        double outputGrad = outputGradient.get(0, outH, outW);
                                        gradSum += kernelVal * outputGrad;
                                    }
                                }
                            }
                        }
                    }
                    
                    inputGradient.set(gradSum, c, inH, inW);
                }
            }
        }
        
        // Compute kernel gradients
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    double gradSum = 0.0;
                    
                    for (int outH = 0; outH < outputHeight; outH++) {
                        for (int outW = 0; outW < outputWidth; outW++) {
                            int inH = outH * stride + kh - padHeight;
                            int inW = outW * stride + kw - padWidth;
                            
                            if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth) {
                                double inputVal = input.get(c, inH, inW);
                                double outputGrad = outputGradient.get(0, outH, outW);
                                gradSum += inputVal * outputGrad;
                            }
                        }
                    }
                    
                    kernelGradient.set(gradSum, c, kh, kw);
                }
            }
        }
        
        return new ConvolutionGradients(inputGradient, kernelGradient);
    }
    
    /**
     * Stack multiple tensors along a specified dimension
     */
    private static Tensor stackTensors(Tensor[] tensors, int dimension) {
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
    
    /**
     * Container for convolution gradients
     */
    public static class ConvolutionGradients {
        public final Tensor inputGradient;
        public final Tensor kernelGradient;
        
        public ConvolutionGradients(Tensor inputGradient, Tensor kernelGradient) {
            this.inputGradient = inputGradient;
            this.kernelGradient = kernelGradient;
        }
    }
}

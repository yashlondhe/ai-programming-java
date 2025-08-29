package com.aiprogramming.ch09;

import com.aiprogramming.ch09.core.RNNCell;
import com.aiprogramming.ch09.core.LSTMCell;
import com.aiprogramming.ch09.core.GRUCell;
import com.aiprogramming.ch09.applications.TextGenerator;
import com.aiprogramming.ch09.applications.SentimentAnalyzer;
import com.aiprogramming.ch09.applications.TimeSeriesPredictor;
import com.aiprogramming.utils.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Main examples demonstrating Recurrent Neural Networks (RNNs), LSTM, and GRU.
 * 
 * This class provides comprehensive examples of:
 * 1. Basic RNN cell usage
 * 2. LSTM cell implementation
 * 3. GRU cell implementation
 * 4. Text generation
 * 5. Sentiment analysis
 * 6. Time series prediction
 */
public class RNNExamples {
    
    public static void main(String[] args) {
        System.out.println("=== Chapter 9: Recurrent Neural Networks (RNNs) ===\n");
        
        // Run basic examples
        basicRNNExample();
        lstmExample();
        gruExample();
        
        // Run application examples
        textGenerationExample();
        sentimentAnalysisExample();
        timeSeriesPredictionExample();
        
        System.out.println("\n=== All examples completed successfully! ===");
    }
    
    /**
     * Example demonstrating basic RNN cell functionality.
     */
    public static void basicRNNExample() {
        System.out.println("1. Basic RNN Cell Example");
        System.out.println("=========================");
        
        // Create RNN cell
        int inputSize = 3;
        int hiddenSize = 4;
        int outputSize = 2;
        RNNCell rnn = new RNNCell(inputSize, hiddenSize, outputSize);
        
        // Create sample sequence
        List<double[]> sequence = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < 5; i++) {
            double[] input = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                input[j] = random.nextGaussian();
            }
            sequence.add(input);
        }
        
        // Validate sequence using utils
        ValidationUtils.validateNotNull(sequence, "sequence");
        ValidationUtils.validateNonEmpty(sequence, "sequence");
        
        // Forward pass
        System.out.println("Input sequence:");
        for (int i = 0; i < sequence.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f, %.3f]\n", 
                i, sequence.get(i)[0], sequence.get(i)[1], sequence.get(i)[2]);
        }
        
        List<double[]> outputs = rnn.forwardSequence(sequence);
        
        System.out.println("\nOutput sequence:");
        for (int i = 0; i < outputs.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f]\n", 
                i, outputs.get(i)[0], outputs.get(i)[1]);
        }
        
        // Demonstrate hidden state
        System.out.printf("\nFinal hidden state: [%.3f, %.3f, %.3f, %.3f]\n",
            rnn.getHiddenState()[0], rnn.getHiddenState()[1], 
            rnn.getHiddenState()[2], rnn.getHiddenState()[3]);
        
        // Calculate output statistics using utils
        double[] outputValues = new double[outputs.size() * 2];
        int idx = 0;
        for (double[] output : outputs) {
            for (double val : output) {
                outputValues[idx++] = val;
            }
        }
        
        System.out.printf("Output Statistics:\n");
        System.out.printf("  Mean: %.3f\n", StatisticsUtils.mean(outputValues));
        System.out.printf("  Standard deviation: %.3f\n", StatisticsUtils.standardDeviation(outputValues));
        
        System.out.println();
    }
    
    /**
     * Example demonstrating LSTM cell functionality.
     */
    public static void lstmExample() {
        System.out.println("2. LSTM Cell Example");
        System.out.println("====================");
        
        // Create LSTM cell
        int inputSize = 2;
        int hiddenSize = 3;
        LSTMCell lstm = new LSTMCell(inputSize, hiddenSize);
        
        // Create sample sequence
        List<double[]> sequence = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 4; i++) {
            double[] input = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                input[j] = random.nextGaussian() * 0.5;
            }
            sequence.add(input);
        }
        
        // Forward pass
        System.out.println("Input sequence:");
        for (int i = 0; i < sequence.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f]\n", 
                i, sequence.get(i)[0], sequence.get(i)[1]);
        }
        
        List<double[]> outputs = lstm.forwardSequence(sequence);
        
        System.out.println("\nHidden state sequence:");
        for (int i = 0; i < outputs.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f, %.3f]\n", 
                i, outputs.get(i)[0], outputs.get(i)[1], outputs.get(i)[2]);
        }
        
        // Demonstrate cell state
        System.out.printf("\nFinal cell state: [%.3f, %.3f, %.3f]\n",
            lstm.getCellState()[0], lstm.getCellState()[1], lstm.getCellState()[2]);
        
        System.out.println();
    }
    
    /**
     * Example demonstrating GRU cell functionality.
     */
    public static void gruExample() {
        System.out.println("3. GRU Cell Example");
        System.out.println("===================");
        
        // Create GRU cell
        int inputSize = 2;
        int hiddenSize = 3;
        GRUCell gru = new GRUCell(inputSize, hiddenSize);
        
        // Create sample sequence
        List<double[]> sequence = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 4; i++) {
            double[] input = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                input[j] = random.nextGaussian() * 0.5;
            }
            sequence.add(input);
        }
        
        // Forward pass
        System.out.println("Input sequence:");
        for (int i = 0; i < sequence.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f]\n", 
                i, sequence.get(i)[0], sequence.get(i)[1]);
        }
        
        List<double[]> outputs = gru.forwardSequence(sequence);
        
        System.out.println("\nHidden state sequence:");
        for (int i = 0; i < outputs.size(); i++) {
            System.out.printf("  Step %d: [%.3f, %.3f, %.3f]\n", 
                i, outputs.get(i)[0], outputs.get(i)[1], outputs.get(i)[2]);
        }
        
        System.out.println();
    }
    
    /**
     * Example demonstrating text generation using RNNs.
     */
    public static void textGenerationExample() {
        System.out.println("4. Text Generation Example");
        System.out.println("==========================");
        
        TextGenerator generator = new TextGenerator();
        
        // Sample text for training
        String trainingText = "The quick brown fox jumps over the lazy dog. " +
                            "This is a sample text for demonstrating text generation. " +
                            "Recurrent neural networks can learn patterns in text data.";
        
        System.out.println("Training text: " + trainingText);
        
        // Train the model (simplified training for demonstration)
        System.out.println("\nTraining text generation model...");
        generator.train(trainingText, 100); // 100 epochs
        
        // Generate text
        String seed = "The quick";
        int length = 50;
        String generatedText = generator.generateText(seed, length);
        
        System.out.println("\nGenerated text (seed: '" + seed + "'):");
        System.out.println(generatedText);
        
        System.out.println();
    }
    
    /**
     * Example demonstrating sentiment analysis using RNNs.
     */
    public static void sentimentAnalysisExample() {
        System.out.println("5. Sentiment Analysis Example");
        System.out.println("=============================");
        
        SentimentAnalyzer analyzer = new SentimentAnalyzer();
        
        // Sample sentences for analysis
        String[] sentences = {
            "I love this movie! It's absolutely fantastic.",
            "This product is terrible and I hate it.",
            "The food was okay, nothing special.",
            "Amazing performance by the actors, highly recommended!",
            "Disappointing experience, would not recommend."
        };
        
        System.out.println("Analyzing sentiment for sample sentences:\n");
        
        for (String sentence : sentences) {
            double sentiment = analyzer.analyzeSentiment(sentence);
            String sentimentLabel = getSentimentLabel(sentiment);
            
            System.out.printf("Text: \"%s\"\n", sentence);
            System.out.printf("Sentiment: %.3f (%s)\n\n", sentiment, sentimentLabel);
        }
        
        System.out.println();
    }
    
    /**
     * Example demonstrating time series prediction using RNNs.
     */
    public static void timeSeriesPredictionExample() {
        System.out.println("6. Time Series Prediction Example");
        System.out.println("=================================");
        
        TimeSeriesPredictor predictor = new TimeSeriesPredictor();
        
        // Generate sample time series data (sine wave with noise)
        double[] historicalData = generateSampleTimeSeries(100);
        
        // Validate data using utils
        ValidationUtils.validateNotNull(historicalData, "historicalData");
        ValidationUtils.validateVector(historicalData, "historicalData");
        
        System.out.println("Sample time series data (first 10 points):");
        for (int i = 0; i < Math.min(10, historicalData.length); i++) {
            System.out.printf("  t=%d: %.3f\n", i, historicalData[i]);
        }
        
        // Calculate time series statistics using utils
        System.out.printf("\nTime Series Statistics:\n");
        System.out.printf("  Mean: %.3f\n", StatisticsUtils.mean(historicalData));
        System.out.printf("  Standard deviation: %.3f\n", StatisticsUtils.standardDeviation(historicalData));
        System.out.printf("  Variance: %.3f\n", StatisticsUtils.variance(historicalData));
        
        // Train the model first
        System.out.println("\nTraining time series prediction model...");
        predictor.train(historicalData, 50); // 50 epochs
        
        // Predict next values
        int predictionLength = 10;
        double[] predictions = predictor.predictNextValues(historicalData, predictionLength);
        
        // Validate predictions using utils
        ValidationUtils.validateNotNull(predictions, "predictions");
        ValidationUtils.validateVector(predictions, "predictions");
        
        System.out.println("\nPredicted values:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.printf("  t=%d: %.3f\n", historicalData.length + i, predictions[i]);
        }
        
        // Calculate prediction statistics using utils
        System.out.printf("\nPrediction Statistics:\n");
        System.out.printf("  Mean: %.3f\n", StatisticsUtils.mean(predictions));
        System.out.printf("  Standard deviation: %.3f\n", StatisticsUtils.standardDeviation(predictions));
        
        System.out.println();
    }
    
    /**
     * Generate sample time series data (sine wave with noise).
     */
    private static double[] generateSampleTimeSeries(int length) {
        double[] data = new double[length];
        Random random = new Random(42);
        
        for (int i = 0; i < length; i++) {
            // Sine wave with some noise
            data[i] = Math.sin(i * 0.1) + random.nextGaussian() * 0.1;
        }
        
        return data;
    }
    
    /**
     * Convert sentiment score to label.
     */
    private static String getSentimentLabel(double sentiment) {
        if (sentiment > 0.6) return "Positive";
        else if (sentiment < 0.4) return "Negative";
        else return "Neutral";
    }
    
    /**
     * Demonstrate gradient clipping for RNN training.
     */
    public static void demonstrateGradientClipping() {
        System.out.println("7. Gradient Clipping Example");
        System.out.println("============================");
        
        // Create RNN cell
        RNNCell rnn = new RNNCell(2, 3, 1);
        
        // Create sample data
        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 5; i++) {
            double[] input = {random.nextGaussian(), random.nextGaussian()};
            double[] target = {random.nextGaussian()};
            inputs.add(input);
            targets.add(target);
        }
        
        // Forward pass
        List<double[]> outputs = rnn.forwardSequence(inputs);
        
        // Compute gradients (simplified)
        List<double[]> gradients = new ArrayList<>();
        for (int i = 0; i < outputs.size(); i++) {
            double[] gradient = new double[1];
            gradient[0] = outputs.get(i)[0] - targets.get(i)[0];
            gradients.add(gradient);
        }
        
        // Apply gradient clipping
        double maxGradNorm = 1.0;
        double totalNorm = 0.0;
        
        for (double[] grad : gradients) {
            for (double g : grad) {
                totalNorm += g * g;
            }
        }
        totalNorm = Math.sqrt(totalNorm);
        
        double clipCoeff = Math.min(1.0, maxGradNorm / totalNorm);
        
        System.out.printf("Original gradient norm: %.3f\n", totalNorm);
        System.out.printf("Clip coefficient: %.3f\n", clipCoeff);
        
        if (clipCoeff < 1.0) {
            System.out.println("Gradients were clipped to prevent explosion.");
        } else {
            System.out.println("Gradients were within normal range.");
        }
        
        System.out.println();
    }
}

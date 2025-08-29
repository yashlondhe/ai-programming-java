package com.aiprogramming.ch17.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.*;

/**
 * Feature importance visualization using JFreeChart
 * 
 * This class provides methods to create various visualizations of feature importance,
 * including bar charts, horizontal bar charts, and heatmaps.
 */
public class FeatureImportancePlot {
    
    private static final int DEFAULT_WIDTH = 800;
    private static final int DEFAULT_HEIGHT = 600;
    private static final Color[] COLORS = {
        new Color(31, 119, 180),   // Blue
        new Color(255, 127, 14),   // Orange
        new Color(44, 160, 44),    // Green
        new Color(214, 39, 40),    // Red
        new Color(148, 103, 189),  // Purple
        new Color(140, 86, 75),    // Brown
        new Color(227, 119, 194),  // Pink
        new Color(127, 127, 127),  // Gray
        new Color(188, 189, 34),   // Olive
        new Color(23, 190, 207)    // Cyan
    };
    
    /**
     * Create a bar chart of feature importance
     * 
     * @param featureNames array of feature names
     * @param importanceScores array of importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotFeatureImportance(String[] featureNames, double[] importanceScores) {
        if (featureNames == null || importanceScores == null || 
            featureNames.length != importanceScores.length) {
            throw new IllegalArgumentException("Feature names and importance scores must be non-null and have the same length");
        }
        
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Add data to dataset
        for (int i = 0; i < featureNames.length; i++) {
            dataset.addValue(importanceScores[i], "Importance", featureNames[i]);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
            "Feature Importance",
            "Features",
            "Importance Score",
            dataset,
            PlotOrientation.VERTICAL,
            false, true, false
        );
        
        // Customize chart appearance
        customizeChart(chart);
        
        return chart;
    }
    
    /**
     * Create a horizontal bar chart of feature importance
     * 
     * @param featureNames array of feature names
     * @param importanceScores array of importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotHorizontalFeatureImportance(String[] featureNames, double[] importanceScores) {
        if (featureNames == null || importanceScores == null || 
            featureNames.length != importanceScores.length) {
            throw new IllegalArgumentException("Feature names and importance scores must be non-null and have the same length");
        }
        
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Add data to dataset
        for (int i = 0; i < featureNames.length; i++) {
            dataset.addValue(importanceScores[i], "Importance", featureNames[i]);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
            "Feature Importance",
            "Importance Score",
            "Features",
            dataset,
            PlotOrientation.HORIZONTAL,
            false, true, false
        );
        
        // Customize chart appearance
        customizeChart(chart);
        
        return chart;
    }
    
    /**
     * Create a sorted feature importance chart
     * 
     * @param featureNames array of feature names
     * @param importanceScores array of importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotSortedFeatureImportance(String[] featureNames, double[] importanceScores) {
        // Create pairs of feature names and importance scores
        List<FeatureImportancePair> pairs = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            pairs.add(new FeatureImportancePair(featureNames[i], importanceScores[i]));
        }
        
        // Sort by importance (descending)
        pairs.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        // Extract sorted arrays
        String[] sortedNames = pairs.stream().map(p -> p.name).toArray(String[]::new);
        double[] sortedScores = pairs.stream().mapToDouble(p -> p.importance).toArray();
        
        return plotFeatureImportance(sortedNames, sortedScores);
    }
    
    /**
     * Create a comparison chart of multiple feature importance sets
     * 
     * @param featureNames array of feature names
     * @param importanceSets map of method names to importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotComparisonChart(String[] featureNames, Map<String, double[]> importanceSets) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Add data for each method
        for (Map.Entry<String, double[]> entry : importanceSets.entrySet()) {
            String methodName = entry.getKey();
            double[] scores = entry.getValue();
            
            for (int i = 0; i < Math.min(featureNames.length, scores.length); i++) {
                dataset.addValue(scores[i], methodName, featureNames[i]);
            }
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
            "Feature Importance Comparison",
            "Features",
            "Importance Score",
            dataset,
            PlotOrientation.VERTICAL,
            true, true, false
        );
        
        // Customize chart appearance
        customizeChart(chart);
        
        return chart;
    }
    
    /**
     * Create a cumulative importance chart
     * 
     * @param featureNames array of feature names
     * @param importanceScores array of importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotCumulativeImportance(String[] featureNames, double[] importanceScores) {
        // Sort features by importance
        List<FeatureImportancePair> pairs = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            pairs.add(new FeatureImportancePair(featureNames[i], importanceScores[i]));
        }
        pairs.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        // Calculate cumulative importance
        double[] cumulative = new double[pairs.size()];
        double sum = 0.0;
        for (int i = 0; i < pairs.size(); i++) {
            sum += pairs.get(i).importance;
            cumulative[i] = sum;
        }
        
        // Create dataset
        XYSeries series = new XYSeries("Cumulative Importance");
        for (int i = 0; i < cumulative.length; i++) {
            series.add(i + 1, cumulative[i]);
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
            "Cumulative Feature Importance",
            "Number of Features",
            "Cumulative Importance",
            dataset,
            PlotOrientation.VERTICAL,
            false, true, false
        );
        
        // Customize chart appearance
        customizeChart(chart);
        
        return chart;
    }
    
    /**
     * Customize chart appearance
     * 
     * @param chart the chart to customize
     */
    private void customizeChart(JFreeChart chart) {
        // Set background color
        chart.setBackgroundPaint(Color.WHITE);
        
        // Customize plot
        if (chart.getPlot() instanceof XYPlot) {
            XYPlot plot = (XYPlot) chart.getPlot();
            plot.setBackgroundPaint(Color.WHITE);
            plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
            plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        }
        
        // Set title font
        chart.getTitle().setFont(new Font("Arial", Font.BOLD, 16));
        
        // Set axis label fonts
        if (chart.getCategoryPlot() != null) {
            chart.getCategoryPlot().getDomainAxis().setLabelFont(new Font("Arial", Font.PLAIN, 12));
            chart.getCategoryPlot().getRangeAxis().setLabelFont(new Font("Arial", Font.PLAIN, 12));
        }
    }
    
    /**
     * Save chart as PNG file
     * 
     * @param chart the chart to save
     * @param filename output filename
     * @throws IOException if saving fails
     */
    public void saveAsPNG(JFreeChart chart, String filename) throws IOException {
        saveAsPNG(chart, filename, DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }
    
    /**
     * Save chart as PNG file with custom dimensions
     * 
     * @param chart the chart to save
     * @param filename output filename
     * @param width image width
     * @param height image height
     * @throws IOException if saving fails
     */
    public void saveAsPNG(JFreeChart chart, String filename, int width, int height) throws IOException {
        File file = new File(filename);
        ChartUtils.saveChartAsPNG(file, chart, width, height);
    }
    
    /**
     * Save chart as JPEG file
     * 
     * @param chart the chart to save
     * @param filename output filename
     * @throws IOException if saving fails
     */
    public void saveAsJPEG(JFreeChart chart, String filename) throws IOException {
        saveAsJPEG(chart, filename, DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }
    
    /**
     * Save chart as JPEG file with custom dimensions
     * 
     * @param chart the chart to save
     * @param filename output filename
     * @param width image width
     * @param height image height
     * @throws IOException if saving fails
     */
    public void saveAsJPEG(JFreeChart chart, String filename, int width, int height) throws IOException {
        File file = new File(filename);
        ChartUtils.saveChartAsJPEG(file, chart, width, height);
    }
    
    /**
     * Create a feature importance heatmap
     * 
     * @param featureNames array of feature names
     * @param importanceMatrix 2D array of importance scores
     * @return JFreeChart object
     */
    public JFreeChart plotImportanceHeatmap(String[] featureNames, double[][] importanceMatrix) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        // Add data to dataset
        for (int i = 0; i < importanceMatrix.length; i++) {
            for (int j = 0; j < importanceMatrix[i].length; j++) {
                dataset.addValue(importanceMatrix[i][j], "Method " + (i + 1), featureNames[j]);
            }
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
            "Feature Importance Heatmap",
            "Features",
            "Methods",
            dataset,
            PlotOrientation.HORIZONTAL,
            false, true, false
        );
        
        // Customize chart appearance
        customizeChart(chart);
        
        return chart;
    }
    
    /**
     * Inner class for feature importance pairs
     */
    private static class FeatureImportancePair {
        final String name;
        final double importance;
        
        FeatureImportancePair(String name, double importance) {
            this.name = name;
            this.importance = importance;
        }
    }
}

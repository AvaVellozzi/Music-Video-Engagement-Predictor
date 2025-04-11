# Music Video Engagement Predictor

This project analyzes YouTube music video data to predict view counts and classify videos into engagement categories.

## Project Structure

- **preprocessing.py**: Contains all the data analysis, preprocessing, and model implementation. This script:
  - Loads and cleans the dataset
  - Performs exploratory data analysis
  - Handles multicollinearity
  - Implements PCA analysis
  - Trains regression models for view count prediction
  - Trains classification models for categorizing videos by view count
  - Visualizes model results and feature importance
  - Saves all outputs to the results directory

- **generate_report.py**: Generates an HTML report from the analysis results.

- **results/**: Directory containing all output files from the analysis, including:
  - Visualizations
  - Model comparison plots
  - Feature importance graphs
  - Decision tree visualizations
  - R² and accuracy plots with standard deviation bars
  - Confusion matrices
  - Model metrics in JSON format

## Running the Analysis

1. Run the preprocessing script: `python preprocessing.py`
2. Generate the HTML report: `python generate_report.py`

The HTML report summarizes all findings and visualizations from the analysis.

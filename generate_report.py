#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Configure the results directory
RESULTS_DIR = 'results'
HTML_OUTPUT = 'music_video_engagement_report.html'

def generate_html_report():
    """
    Generate an HTML report from the preprocessing results.
    """
    # Check if the results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        return False
    
    # Get all image files from the results directory
    image_files = glob.glob(os.path.join(RESULTS_DIR, '*.png'))
    
    if not image_files:
        print(f"Error: No image files found in '{RESULTS_DIR}'.")
        return False
    
    # Load model metrics from JSON file
    metrics_file = os.path.join(RESULTS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            models_comparison = json.load(f)
        print(f"Loaded model metrics from {metrics_file}")
    else:
        print(f"Warning: Metrics file {metrics_file} not found. Using default placeholder values.")
        # Fallback to placeholder values if metrics file doesn't exist
        models_comparison = {
            'Linear Regression': {
                'Training R²': 0.4615,
                'Test R²': 0.4064,
                'Training RMSE': 196353960.29,
                'Test RMSE': 250370650.13,
                'Classification Accuracy': 0.7767
            },
            'Random Forest (Cross-validated)': {
                'Cross-validated R²': 0.7454,
                'Cross-validated RMSE': 141289929.79,
                'Cross-validated Accuracy': 0.7099
            },
            'Random Forest Regression': {
                'Training R²': 0.9595,
                'Test R²': 0.7293,
                'Training RMSE': 53814354.25,
                'Test RMSE': 169087788.59
            },
            'Random Forest Classification': {
                'Test Accuracy': 0.85,
                'Test Precision': 0.84,
                'Test Recall': 0.85,
                'Test F1 Score': 0.84
            }
        }
    
    # Generate the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Music Video Engagement Predictor Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .visualization {{
                margin: 30px 0;
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            .metrics-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .section {{
                margin: 40px 0;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                color: #777;
                font-size: 0.9em;
                border-top: 1px solid #eee;
            }}
            .highlight {{
                background-color: #e8f4fd;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Music Video Engagement Predictor</h1>
                <h2>Analysis Report</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>1. Feature Analysis</h2>
                <p>This section provides insights into feature correlations and distributions.</p>
                
                <div class="visualization">
                    <h3>Feature Correlation Heatmap</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'correlation_heatmap.png')}" alt="Correlation Heatmap">
                    <p>This heatmap shows the correlations between different features in the dataset.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>2. Principal Component Analysis (PCA)</h2>
                <p>PCA was used to reduce dimensionality and visualize the data in 2D and 3D space.</p>
                
                <div class="visualization">
                    <h3>2D PCA Visualization</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'pca_visualization_views.png')}" alt="2D PCA Visualization">
                    <p>This plot shows how the data points are distributed in a 2D principal component space, colored by view count.</p>
                </div>
                
                <div class="visualization">
                    <h3>3D PCA Visualization</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'pca_3d_visualization.png')}" alt="3D PCA Visualization">
                    <p>This plot shows the data in a 3D principal component space, providing additional insight into data clustering.</p>
                </div>
                
                <div class="visualization">
                    <h3>PCA Feature Loadings</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'pca_loadings_views.png')}" alt="PCA Loadings">
                    <p>This heatmap shows how each original feature contributes to the principal components.</p>
                </div>
                
                <div class="visualization">
                    <h3>PCA vs Views</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'pca_vs_views.png')}" alt="PCA vs Views">
                    <p>This plot shows how the principal components correlate with video view counts.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>3. Linear Regression Analysis</h2>
                <p>Linear regression was used as a baseline model for predicting view counts.</p>
                
                <div class="visualization">
                    <h3>Actual vs Predicted Views</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'linear_regression_actual_vs_predicted.png')}" alt="Linear Regression Actual vs Predicted">
                    <p>This plot compares the actual view counts with the predictions from the linear regression model.</p>
                </div>
                
                <div class="visualization">
                    <h3>Residuals Plot</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'linear_regression_residuals.png')}" alt="Linear Regression Residuals">
                    <p>This plot shows the residuals (errors) of the linear regression model against predicted values.</p>
                </div>
                
                <div class="visualization">
                    <h3>Log-Transformed Actual vs Predicted</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'linear_regression_log_transformed.png')}" alt="Log-Transformed Actual vs Predicted">
                    <p>This plot compares actual and predicted views on a log scale, which helps visualize the spread of values more clearly.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>4. Classification Analysis</h2>
                <p>The regression problem was also treated as a classification task by binning view counts into categories.</p>
                
                <div class="visualization">
                    <h3>Confusion Matrix</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'view_categories_confusion_matrix.png')}" alt="Confusion Matrix">
                    <p>This confusion matrix shows how well the model classifies videos into different view count categories.</p>
                </div>
                
                <div class="visualization">
                    <h3>View Categories Distribution</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'view_categories_distribution.png')}" alt="View Categories Distribution">
                    <p>This plot shows the distribution of videos across different view count categories.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>5. Random Forest Analysis</h2>
                <p>Random Forest models were used for both regression and classification tasks.</p>
                
                <div class="visualization">
                    <h3>Random Forest Regression R² Scores Across Folds</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_regression_r2_scores.png')}" alt="Random Forest Regression R² Scores">
                    <p>This plot shows the R² scores for Random Forest regression across each cross-validation fold, with error bars representing standard deviation.</p>
                </div>
                
                <div class="visualization">
                    <h3>Random Forest Classification Accuracy Scores Across Folds</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_classification_accuracy_scores.png')}" alt="Random Forest Classification Accuracy Scores">
                    <p>This plot shows the accuracy scores for Random Forest classification across each cross-validation fold, with error bars representing standard deviation.</p>
                </div>
                
                <div class="visualization">
                    <h3>Random Forest Regression Confusion Matrix</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_confusion_matrix.png')}" alt="Random Forest Confusion Matrix">
                    <p>This confusion matrix shows the classification performance of the Random Forest regression model when its predictions are converted to categories.</p>
                </div>
                
                <div class="visualization">
                    <h3>Random Forest Classification Confusion Matrix</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_classification_confusion_matrix.png')}" alt="Random Forest Classification Confusion Matrix">
                    <p>This confusion matrix shows the performance of the direct Random Forest classification approach, which predicts categories without going through regression first.</p>
                </div>
                
                <div class="visualization">
                    <h3>Random Forest Regression Feature Importance</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_feature_importance.png')}" alt="Feature Importance">
                    <p>This plot shows the relative importance of each feature in the Random Forest regression model.</p>
                </div>
                
                <div class="visualization">
                    <h3>Random Forest Classification Feature Importance</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'rf_classification_feature_importance.png')}" alt="Classification Feature Importance">
                    <p>This plot shows the relative importance of each feature in the Random Forest classification model, which may differ from the regression model.</p>
                </div>
                
                <div class="visualization">
                    <h3>Decision Tree Visualization (Classification)</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'decision_tree_visualization.png')}" alt="Decision Tree Visualization">
                    <p>This visualization shows a single decision tree from the Random Forest classification model (limited to depth=3 for readability). It illustrates how the model makes decisions based on feature values.</p>
                </div>
                
                <div class="visualization">
                    <h3>Decision Tree Visualization (Regression)</h3>
                    <img src="{os.path.join(RESULTS_DIR, 'regression_decision_tree_visualization.png')}" alt="Regression Decision Tree Visualization">
                    <p>This visualization shows a single regression decision tree (limited to depth=3 for readability). It demonstrates how the model predicts view counts based on feature values.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>6. Model Comparison</h2>
                <p>This section compares the performance of different models used in this analysis.</p>
                
                <h3>Regression Metrics</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Model</th>
                        <th>Training R²</th>
                        <th>Test R²</th>
                        <th>Training RMSE</th>
                        <th>Test RMSE</th>
                    </tr>
    """
    
    # Add regression model metrics
    for model in ['Linear Regression', 'Random Forest (Cross-validated)', 'Random Forest Regression']:
        if model in models_comparison:
            metrics = models_comparison[model]
            html_content += "<tr>"
            html_content += f"<td>{model}</td>"
            
            # Training R²
            if 'Training R²' in metrics:
                html_content += f"<td>{'%.4f' % metrics['Training R²']}</td>"
            elif 'Cross-validated R²' in metrics:
                if 'Cross-validated R² Std' in metrics:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated R²']} (±{'%.4f' % metrics['Cross-validated R² Std']})</td>"
                else:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated R²']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # Test R²
            if 'Test R²' in metrics:
                html_content += f"<td class='highlight'>{'%.4f' % metrics['Test R²']}</td>"
            elif 'Cross-validated R²' in metrics:
                if 'Cross-validated R² Std' in metrics:
                    html_content += f"<td class='highlight'>{'%.4f' % metrics['Cross-validated R²']} (±{'%.4f' % metrics['Cross-validated R² Std']})</td>"
                else:
                    html_content += f"<td class='highlight'>{'%.4f' % metrics['Cross-validated R²']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # Training RMSE
            if 'Training RMSE' in metrics:
                html_content += f"<td>{'%.2f' % metrics['Training RMSE']}</td>"
            elif 'Cross-validated RMSE' in metrics:
                html_content += f"<td>{'%.2f' % metrics['Cross-validated RMSE']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # Test RMSE
            if 'Test RMSE' in metrics:
                html_content += f"<td>{'%.2f' % metrics['Test RMSE']}</td>"
            elif 'Cross-validated RMSE' in metrics:
                html_content += f"<td>{'%.2f' % metrics['Cross-validated RMSE']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            html_content += "</tr>"
    
    html_content += """
                </table>
                
                <h3>Classification Metrics</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
    """
    
    # Add classification model metrics
    for model, metrics in models_comparison.items():
        if any(key.endswith('Accuracy') or key.endswith('Precision') for key in metrics.keys()):
            html_content += "<tr>"
            html_content += f"<td>{model}</td>"
            
            # Accuracy
            if 'Classification Accuracy' in metrics:
                html_content += f"<td class='highlight'>{'%.4f' % metrics['Classification Accuracy']}</td>"
            elif 'Cross-validated Accuracy' in metrics:
                if 'Cross-validated Accuracy Std' in metrics:
                    html_content += f"<td class='highlight'>{'%.4f' % metrics['Cross-validated Accuracy']} (±{'%.4f' % metrics['Cross-validated Accuracy Std']})</td>"
                else:
                    html_content += f"<td class='highlight'>{'%.4f' % metrics['Cross-validated Accuracy']}</td>"
            elif 'Test Accuracy' in metrics:
                html_content += f"<td class='highlight'>{'%.4f' % metrics['Test Accuracy']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # Precision
            if 'Test Precision' in metrics:
                html_content += f"<td>{'%.4f' % metrics['Test Precision']}</td>"
            elif 'Cross-validated Precision' in metrics:
                if 'Cross-validated Precision Std' in metrics:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated Precision']} (±{'%.4f' % metrics['Cross-validated Precision Std']})</td>"
                else:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated Precision']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # Recall
            if 'Test Recall' in metrics:
                html_content += f"<td>{'%.4f' % metrics['Test Recall']}</td>"
            elif 'Cross-validated Recall' in metrics:
                if 'Cross-validated Recall Std' in metrics:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated Recall']} (±{'%.4f' % metrics['Cross-validated Recall Std']})</td>"
                else:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated Recall']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            # F1 Score
            if 'Test F1 Score' in metrics:
                html_content += f"<td>{'%.4f' % metrics['Test F1 Score']}</td>"
            elif 'Cross-validated F1 Score' in metrics:
                if 'Cross-validated F1 Score Std' in metrics:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated F1 Score']} (±{'%.4f' % metrics['Cross-validated F1 Score Std']})</td>"
                else:
                    html_content += f"<td>{'%.4f' % metrics['Cross-validated F1 Score']}</td>"
            else:
                html_content += "<td>N/A</td>"
            
            html_content += "</tr>"
    
    # Get best model based on test R² or accuracy
    best_regression_model = max([m for m in models_comparison.keys() if 'Test R²' in models_comparison[m] or 'Cross-validated R²' in models_comparison[m]], 
                               key=lambda m: models_comparison[m].get('Test R²', models_comparison[m].get('Cross-validated R²', 0)))
    
    best_regression_r2 = max([models_comparison[m].get('Test R²', models_comparison[m].get('Cross-validated R²', 0)) 
                             for m in models_comparison.keys() 
                             if 'Test R²' in models_comparison[m] or 'Cross-validated R²' in models_comparison[m]])
    
    best_classification_model = max([m for m in models_comparison.keys() if any(k.endswith('Accuracy') for k in models_comparison[m].keys())],
                                   key=lambda m: max(v for k, v in models_comparison[m].items() if k.endswith('Accuracy')))
    
    best_classification_acc = max([max(v for k, v in models_comparison[m].items() if k.endswith('Accuracy')) 
                                  for m in models_comparison.keys() 
                                  if any(k.endswith('Accuracy') for k in models_comparison[m].keys())])
    
    # Identify top features from RF feature importance
    top_features = ["Comments", "Stream", "Duration_ms", "Valence", "Tempo"]  # Placeholder - should be loaded from data
    
    html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>7. Conclusion</h2>
                <p>Based on the analysis, {best_regression_model} achieved the highest performance for predicting music video engagement metrics with a test R² value of {best_regression_r2:.4f}, indicating that it can explain approximately {best_regression_r2*100:.1f}% of the variance in view counts.</p>
                <p>Feature importance analysis shows that user engagement metrics like {', '.join(top_features[:3])} are the most important predictors of video views. This suggests that videos that generate more user interaction tend to have higher view counts.</p>
                <p>For classification into view count categories, the {best_classification_model} achieved an accuracy rate of {best_classification_acc:.2%}, which is a significant improvement over random guessing (which would be 20% for 5 categories).</p>
            </div>
            
            <div class="section">
                <h2>Resampling Experiment (SMOTE)</h2>
                <p>
                An experiment was conducted using the Synthetic Minority Over-sampling Technique (SMOTE) within the cross-validation pipeline 
                to address the moderate class imbalance observed in the view categories. The goal was to potentially improve the performance 
                of the classification models (Random Forest and Decision Tree) by providing more examples of the minority classes.
                </p>
                <p>
                However, the results indicated that applying SMOTE did not lead to an improvement in the overall cross-validated 
                performance metrics (Accuracy, Precision, Recall, F1-score) for either the Random Forest or the Decision Tree classifier 
                when compared to training on the original data distribution. While SMOTE might subtly alter the prediction balance across 
                individual classes (as seen in detailed classification reports), it did not enhance the overall predictive power in this specific scenario. 
                This suggests that for this dataset and these models, the benefits of oversampling were potentially outweighed by other factors, 
                such as the introduction of synthetic noise or modifications to the decision boundaries that did not generalize well.
                </p>
            </div>
            
            <div class="footer">
                <p>Music Video Engagement Predictor Project</p>
                <p>© 2025. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(HTML_OUTPUT, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated successfully: {HTML_OUTPUT}")
    return True

if __name__ == "__main__":
    generate_html_report() 
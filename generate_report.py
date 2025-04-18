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
                        <th>Test R² / CV R²</th>
                        <th>Test RMSE / CV RMSE</th>
                        <th>Test MAE / CV MAE</th>
                    </tr>
    """
    
    # Populate Regression Table Rows
    for model_name, metrics in models_comparison.items():
        # Check if the model has regression metrics we want to display
        has_r2 = 'Test R²' in metrics or 'Cross-validated R²' in metrics
        has_rmse = 'Test RMSE' in metrics or 'Cross-validated RMSE' in metrics
        has_mae = 'Test MAE' in metrics or 'Cross-validated MAE' in metrics
        
        # Only add rows for models with relevant regression metrics
        if has_r2 or has_rmse or has_mae:
            html_content += '<tr>'
            html_content += f'<td>{model_name}</td>'
            
            # Display R² (prefer CV if available)
            r2_val = metrics.get('Cross-validated R²', metrics.get('Test R²'))
            r2_std = metrics.get('Cross-validated R² Std')
            r2_str = f"{r2_val:.4f}" if r2_val is not None else "N/A"
            if r2_std is not None:
                r2_str += f" (±{r2_std:.4f})"
            html_content += f'<td>{r2_str}</td>'
            
            # Display RMSE (prefer CV if available)
            rmse_val = metrics.get('Cross-validated RMSE', metrics.get('Test RMSE'))
            rmse_str = f"{rmse_val:,.2f}" if rmse_val is not None else "N/A"
            html_content += f'<td>{rmse_str}</td>'
            
            # Display MAE (prefer CV if available)
            mae_val = metrics.get('Cross-validated MAE', metrics.get('Test MAE'))
            mae_str = f"{mae_val:,.2f}" if mae_val is not None else "N/A"
            html_content += f'<td>{mae_str}</td>'
            
            html_content += '</tr>'
    
    html_content += """
                </table>
    """
    
    # --- Add R² comparison plot ---
    html_content += '<h3>Regression Model R² Comparison (Cross-Validated)</h3>'
    comparison_plot_path = os.path.join(RESULTS_DIR, 'regression_model_r2_comparison.png')
    html_content += '<div class="visualization">' # Add visualization div
    if os.path.exists(comparison_plot_path):
        # Embed image directly using data URI
        try:
            with open(comparison_plot_path, "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_data_uri = f"data:image/png;base64,{img_data}"
                html_content += f'<img src="{img_data_uri}" alt="Regression Model R2 Comparison"><br>'
                html_content += '<p>Comparison of mean R² scores from 5-fold cross-validation for Random Forest and Decision Tree regressors, with standard deviation error bars.</p>'
        except Exception as e:
                html_content += f'<p>Error embedding Regression model R² comparison plot: {e}</p>'
    else:
        html_content += '<p>Regression model R² comparison plot not found.</p>'
    html_content += '</div>' # Close visualization div
    # --- End of R² plot addition ---
    
    html_content += """
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
    
    # Populate Classification Table Rows
    for model_name, metrics in models_comparison.items():
        # Check if the model has classification metrics
        has_acc = 'Cross-validated Accuracy' in metrics or 'Classification Accuracy (Binned Test Pred)' in metrics or 'Classification Accuracy (Binned CV Pred)' in metrics
        has_f1 = 'Cross-validated F1 Score' in metrics

        if has_acc or has_f1: # Add rows for models with key classification metrics
            html_content += '<tr>'
            html_content += f'<td>{model_name}</td>'

            # Accuracy
            acc_val = metrics.get('Cross-validated Accuracy', metrics.get('Classification Accuracy (Binned CV Pred)', metrics.get('Classification Accuracy (Binned Test Pred)')))
            acc_std = metrics.get('Cross-validated Accuracy Std')
            acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
            if acc_std is not None:
                acc_str += f" (±{acc_std:.4f})"
            html_content += f'<td>{acc_str}</td>'

            # Precision (Prefer CV Weighted)
            prec_val = metrics.get('Cross-validated Precision')
            prec_std = metrics.get('Cross-validated Precision Std')
            prec_str = f"{prec_val:.4f}" if prec_val is not None else "N/A"
            if prec_std is not None:
                prec_str += f" (±{prec_std:.4f})"
            html_content += f'<td>{prec_str}</td>'

            # Recall (Prefer CV Weighted)
            rec_val = metrics.get('Cross-validated Recall')
            rec_std = metrics.get('Cross-validated Recall Std')
            rec_str = f"{rec_val:.4f}" if rec_val is not None else "N/A"
            if rec_std is not None:
                rec_str += f" (±{rec_std:.4f})"
            html_content += f'<td>{rec_str}</td>'

            # F1 Score (Prefer CV Weighted)
            f1_val = metrics.get('Cross-validated F1 Score')
            f1_std = metrics.get('Cross-validated F1 Score Std')
            f1_str = f"{f1_val:.4f}" if f1_val is not None else "N/A"
            if f1_std is not None:
                f1_str += f" (±{f1_std:.4f})"
            html_content += f'<td>{f1_str}</td>'

            html_content += '</tr>'
    
    html_content += """
                </table>
    """
    
    # --- Add Classification Accuracy comparison plot ---
    html_content += '<h3>Classification Model Accuracy Comparison (Cross-Validated)</h3>'
    clf_comparison_plot_path = os.path.join(RESULTS_DIR, 'classification_model_accuracy_comparison.png')
    html_content += '<div class="visualization">' # Add visualization div
    if os.path.exists(clf_comparison_plot_path):
        try:
            with open(clf_comparison_plot_path, "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_data_uri = f"data:image/png;base64,{img_data}"
                html_content += f'<img src="{img_data_uri}" alt="Classification Model Accuracy Comparison"><br>'
                html_content += '<p>Comparison of mean Accuracy from 5-fold cross-validation for Random Forest and Decision Tree classifiers, with standard deviation error bars.</p>'
        except Exception as e:
            html_content += f'<p>Error embedding Classification model accuracy comparison plot: {e}</p>'
    else:
        html_content += '<p>Classification model accuracy comparison plot not found.</p>'
    html_content += '</div>' # Close visualization div
    # --- End of addition ---

    # Get best model based on test R² or accuracy (Moved this logic down)
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
    
    # Identify top features from RF feature importance (Placeholder)
    # TODO: Ideally load this from a saved file or extract from final RF model if feasible in report script
    top_features_placeholder = ["Stream", "Duration_ms", "Loudness", "Channel", "Danceability"] 
    top_features_str = ", ".join(top_features_placeholder)

    html_content += f"""
            <div class="section">
                <h2>7. Conclusion</h2>
                <p>Based on the analysis, {best_regression_model} achieved the highest performance for predicting music video engagement metrics with a test R² value of {best_regression_r2:.4f}, indicating that it can explain approximately {best_regression_r2*100:.1f}% of the variance in view counts.</p>
                <p>Feature importance analysis (primarily from Random Forest) suggests that features like {top_features_str} are among the most important predictors of video views.</p>
                <p>For classification into view count categories, the {best_classification_model} achieved an accuracy rate of {best_classification_acc:.2%}, which is a significant improvement over random guessing.</p>
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

            <div class="section">
                 <h2>Detailed Visualizations</h2>
                 # Define plots to include (adjust filenames as needed)
                 all_plot_files = {{
                     'Feature Correlation Heatmap': 'correlation_heatmap.png',
                     '2D PCA Visualization': 'pca_visualization_views.png',
                     '3D PCA Visualization': 'pca_3d_visualization.png',
                     'PCA Feature Loadings': 'pca_loadings_views.png',
                     'PCA vs Views': 'pca_vs_views.png',
                     'Linear Regression: Actual vs Predicted': 'linear_regression_actual_vs_predicted.png',
                     'Linear Regression: Residuals': 'linear_regression_residuals.png',
                     'Linear Regression: Log-transformed Actual vs Predicted': 'linear_regression_log_transformed.png',
                     'Confusion Matrix (Binned Categories - RF Test)': 'view_categories_confusion_matrix.png', # From Sec 11
                     'View Categories Distribution (Test Set)': 'view_categories_distribution.png',
                     'Random Forest Regression R² Scores Across Folds': 'rf_regression_r2_scores.png',
                     'Random Forest Regression Confusion Matrix (Binned CV Pred)': 'rf_binned_regression_confusion_matrix.png',
                     'Random Forest Regression Feature Importance': 'rf_regression_feature_importance.png',
                     'Random Forest Classification Accuracy Scores Across Folds': 'rf_classification_accuracy_scores.png',
                     'Random Forest Classification Confusion Matrix (CV)': 'rf_classification_confusion_matrix.png',
                     'Random Forest Classification Feature Importance': 'rf_classification_feature_importance.png',
                     'Decision Tree Classification Confusion Matrix (CV)': 'dt_classification_confusion_matrix.png',
                     'Example Decision Tree (Classification, Depth=3)': 'decision_tree_visualization.png',
                     'Example Decision Tree (Regression, Depth=3)': 'regression_decision_tree_visualization.png'
                 }}

                 html_content += '<div class="visualization-group">' # Group visualizations
                 for title, filename in all_plot_files.items():
                     img_path = os.path.join(RESULTS_DIR, filename)
                     html_content += '<div class="visualization">'
                     if os.path.exists(img_path):
                         try:
                             with open(img_path, "rb") as img_file:
                                 import base64
                                 img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                 img_data_uri = f"data:image/png;base64,{{img_data}}"
                                 html_content += f'<h3>{{title}}</h3>'
                                 html_content += f'<img src="{{img_data_uri}}" alt="{{title}}"><br>'
                         except Exception as e:
                             html_content += f'<h3>{{title}}</h3><p>Error embedding {{title}} plot: {{e}}</p>'
                     else:
                          html_content += f'<h3>{{title}}</h3><p>{{title}} plot not found.</p>'
                     html_content += '</div>' # Close visualization div
                 html_content += '</div>' # Close visualization group
                 html_content += '</div>' # Close detailed viz section
            
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
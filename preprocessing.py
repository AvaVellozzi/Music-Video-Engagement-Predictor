# ==============================================================================
# Section 1: Imports
# ==============================================================================
import pandas as pd
# Removed unused imports: pandas.plotting, matplotlib.pyplot, numpy, sklearn
# Re-added necessary imports based on the code below:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys # For exit on error
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier

# ==============================================================================
# Create results directory if it doesn't exist
# ==============================================================================
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory: {RESULTS_DIR}")
else:
    print(f"Results will be saved to existing directory: {RESULTS_DIR}")

# ==============================================================================
# Section 2: Configuration
# ==============================================================================
FILEPATH = 'Spotify_Youtube.csv'
TARGET_COLUMN = 'Views' # Target for regression task
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==============================================================================
# Section 3: Load Data
# ==============================================================================
print(f"--- 3: Loading Data from {FILEPATH} ---")
df = pd.read_csv(FILEPATH)
print(f"Data loaded successfully. Shape: {df.shape}")

print("\n--- Initial Data Info ---")
print(df.info())
print("\n--- Data Head ---")
print(df.head())

# ==============================================================================
# Section 4: Initial Cleaning
# ==============================================================================
print("\n--- 4: Cleaning Data ---")

# --- 4a: Drop Unnamed Column ---
if df.columns[0].startswith('Unnamed:'):
    print(f"Dropping the first unnamed column: {df.columns[0]}")
    df = df.iloc[:, 1:]
else:
    print("First column is not unnamed, not dropping.")

# --- 4b: Handle Missing Values ---
initial_rows = len(df)
df.dropna(inplace=True)
rows_dropped = initial_rows - len(df)
if rows_dropped > 0:
    print(f"Dropped {rows_dropped} rows with missing values.")
else:
    print("No rows with missing values found.")

# --- 4c: Remove Outliers (Videos with > 1 billion views) ---
initial_rows = len(df)
outlier_threshold = 1_000_000_000  # 1 billion views
outliers = df[df['Views'] > outlier_threshold]
df = df[df['Views'] <= outlier_threshold]
outliers_dropped = initial_rows - len(df)
print(f"Dropped {outliers_dropped} rows with more than {outlier_threshold:,} views (outliers).")

print(f"DataFrame shape after cleaning: {df.shape}")

# --- 4d: Placeholder for other outlier handling ---
print("\nPlaceholder: Additional outlier detection and handling needed.")

# --- 4e: Placeholder for Incorrectly Labeled Points ---
print("Placeholder: Handling of incorrectly labeled points needed.")


# ==============================================================================
# Section 5: Encode Categorical Features
# ==============================================================================
print("\n--- 5: Encoding Categorical Features ---")
label_encoders = {}

# Get all object columns that need encoding
object_columns = df.select_dtypes(include=['object']).columns
print(f"Found {len(object_columns)} categorical columns that need encoding")

# Encode all object/categorical columns
for col in object_columns:
    if col in df.columns:
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le # Store encoder if needed later
    else:
        print(f"Warning: Column '{col}' not found for encoding.")
        
print(f"DataFrame shape after encoding: {df.shape}")
print("All categorical columns have been encoded to numeric values")

# ==============================================================================
# Section 6: Feature Selection
# ==============================================================================
print("\n--- 6: Feature Selection ---")

# Drop columns that don't add value for ML predictions
columns_to_drop = [
    'Url_spotify',
    'Uri',
    'Url_youtube',
    'Description',
    'Licensed',
    'official_video',
    'Track',  # Dropping Track feature as requested
    'Comments'  # Dropping Comments feature as requested
]

# Drop the specified columns
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")
print(f"DataFrame shape after feature selection: {df.shape}")

# Verify that Views exists in the DataFrame
if TARGET_COLUMN in df.columns:
    print(f"Target column '{TARGET_COLUMN}' is present in DataFrame.")
else:
    print(f"WARNING: Target column '{TARGET_COLUMN}' not found in DataFrame.")
    sys.exit(1)

# Check for multicollinearity (features with high correlation > 0.8)
print("\nChecking for multicollinearity (corr > 0.8):")
high_corr = []
corr_matrix = df.corr()  # Calculate the correlation matrix

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            high_corr.append((col_i, col_j, corr_matrix.iloc[i, j]))

# Drop features with high multicollinearity, but preserve the target column
if high_corr:
    print("Features with high correlation (potential multicollinearity):")
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} & {feat2}: {corr:.2f}")
        # Don't drop the target column (Views)
        if feat1 == TARGET_COLUMN:
            print(f"  Not dropping {feat1} as it is the target variable")
            # Drop the other correlated feature instead
            df = df.drop(columns=[feat2])
            print(f"  Dropped {feat2} instead")
        elif feat2 == TARGET_COLUMN:
            print(f"  Not dropping {feat2} as it is the target variable")
            # Drop the other correlated feature instead
            df = df.drop(columns=[feat1])
            print(f"  Dropped {feat1} instead")
        else:
            # If neither is the target, drop the second feature as before
            df = df.drop(columns=[feat2])
            print(f"  Dropped {feat2}")

    print(f"Dropped features due to multicollinearity. New shape: {df.shape}")
else:
    print("No high correlation detected between features")

# Verify again that Views is still in the DataFrame
if TARGET_COLUMN in df.columns:
    print(f"Target column '{TARGET_COLUMN}' is still present in DataFrame after multicollinearity handling.")
else:
    print(f"ERROR: Target column '{TARGET_COLUMN}' was dropped during multicollinearity handling.")
    sys.exit(1)

# Display remaining columns
print("\nRemaining columns:")
print(df.columns.tolist())

# ==============================================================================
# Section 6.5: Feature Correlation Analysis
# ==============================================================================
print("\n--- 6.5: Feature Correlation Analysis ---")

# Create a correlation matrix for numerical features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
plt.close()
print(f"Correlation heatmap saved to {RESULTS_DIR}/correlation_heatmap.png")

# Display top correlations with the target variable
if TARGET_COLUMN in numeric_df.columns:
    target_correlations = corr_matrix[TARGET_COLUMN].sort_values(ascending=False)
    print(f"\nTop correlations with {TARGET_COLUMN}:")
    print(target_correlations)
else:
    print(f"\nTarget column {TARGET_COLUMN} not found in numeric columns")

# Check for multicollinearity (features with high correlation > 0.8)
print("\nChecking for multicollinearity (corr > 0.8):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr:
    print("Features with high correlation (potential multicollinearity):")
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} & {feat2}: {corr:.2f}")
else:
    print("No high correlation detected between features")

# ==============================================================================
# Section 6.6: PCA (Principal Component Analysis)
# ==============================================================================
print("\n--- 6.6: PCA Analysis for View Prediction ---")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# We need to work with only numerical features for PCA (excluding our target variable Views)
numeric_features = df.select_dtypes(include=['float64', 'int64'])

# Make sure to remove the target variable (Views) from the features for PCA
if TARGET_COLUMN in numeric_features.columns:
    target_views = numeric_features[TARGET_COLUMN].copy()
    numeric_features = numeric_features.drop(columns=[TARGET_COLUMN])
    print(f"Target variable '{TARGET_COLUMN}' separated for visualization with PCA features")
else:
    print(f"Warning: Target variable '{TARGET_COLUMN}' not found in numeric features")
    # If we don't have Views in our dataframe at this point, something went wrong
    target_views = None

# Standardize the features (important for PCA)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)
print(f"Features standardized, shape: {scaled_features.shape}")

# Apply PCA with 2 components for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
print(f"PCA applied, reduced to {pca_result.shape[1]} dimensions")

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Add the target variable for coloring (only if we have it)
if target_views is not None:
    # For better visualization with a skewed target, we can log-transform Views
    log_views = np.log10(target_views + 1)  # Add 1 to avoid log(0)
    pca_df['Log_Views'] = log_views
    print(f"Added log-transformed {TARGET_COLUMN} for visualization")

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: PC1 = {explained_variance[0]:.4f}, PC2 = {explained_variance[1]:.4f}")
print(f"Total variance explained: {sum(explained_variance):.4f}")

# Create a scatter plot of PCA results (2D)
plt.figure(figsize=(10, 8))
if target_views is not None:
    # Color by log-transformed Views
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                       c=pca_df['Log_Views'], 
                       cmap='viridis', 
                       alpha=0.6)
    plt.colorbar(scatter, label=f'Log10({TARGET_COLUMN})')
    plt.title(f'PCA: Feature Projection with {TARGET_COLUMN} (log-scale) as Color')
else:
    # Simple scatter plot without target coloring
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    plt.title('PCA: 2-Component Projection (without target variable)')

plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'pca_visualization_views.png'))
plt.close()
print(f"PCA visualization saved to {RESULTS_DIR}/pca_visualization_views.png")

# 3D PCA Visualization with 3 components
print("\n--- Creating 3D PCA Visualization ---")
# Rerun PCA with 3 components
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(scaled_features)
explained_variance_3d = pca_3d.explained_variance_ratio_
print(f"PCA 3D applied, reduced to 3 dimensions")
print(f"Explained variance ratio: PC1={explained_variance_3d[0]:.4f}, PC2={explained_variance_3d[1]:.4f}, PC3={explained_variance_3d[2]:.4f}")
print(f"Total variance explained by 3 components: {sum(explained_variance_3d):.4f}")

# Create a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

if target_views is not None:
    # Create a color map based on log-transformed views
    log_views = np.log10(target_views + 1)
    scatter = ax.scatter(
        pca_result_3d[:, 0],  # PC1
        pca_result_3d[:, 1],  # PC2
        pca_result_3d[:, 2],  # PC3
        c=log_views,
        cmap='viridis',
        alpha=0.6,
        s=30  # Point size
    )
    fig.colorbar(scatter, ax=ax, label=f'Log10({TARGET_COLUMN})')
else:
    ax.scatter(
        pca_result_3d[:, 0],
        pca_result_3d[:, 1],
        pca_result_3d[:, 2],
        alpha=0.6,
        s=30
    )

ax.set_xlabel(f'PC1 ({explained_variance_3d[0]:.1%})')
ax.set_ylabel(f'PC2 ({explained_variance_3d[1]:.1%})')
ax.set_zlabel(f'PC3 ({explained_variance_3d[2]:.1%})')
ax.set_title('3D PCA Projection of Features')

# Add a grid
ax.grid(True)

# Save the figure
plt.savefig(os.path.join(RESULTS_DIR, 'pca_3d_visualization.png'))
plt.close()
print(f"3D PCA visualization saved to {RESULTS_DIR}/pca_3d_visualization.png")

# Feature importance in PCA
component_names = [f"PC{i+1}" for i in range(2)]
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=component_names, 
    index=numeric_features.columns
)

# Display feature contributions to PCA components
print("\nFeature contributions to principal components:")
print(loadings)

# Plot the feature loadings
plt.figure(figsize=(12, 10))
loadings_plot = sns.heatmap(loadings, cmap='coolwarm', annot=True, fmt=".3f")
plt.title('PCA Feature Loadings for View Prediction')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_loadings_views.png'))
plt.close()
print(f"PCA loadings heatmap saved to {RESULTS_DIR}/pca_loadings_views.png")

# Add a visualization to show how PC1 and PC2 correlate with Views
if target_views is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PC1 vs Views
    ax1.scatter(pca_df['PC1'], target_views, alpha=0.5, c='royalblue')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel(TARGET_COLUMN)
    ax1.set_title(f'PC1 vs {TARGET_COLUMN}')
    ax1.grid(alpha=0.3)
    
    # PC2 vs Views
    ax2.scatter(pca_df['PC2'], target_views, alpha=0.5, c='forestgreen')
    ax2.set_xlabel('Principal Component 2')
    ax2.set_ylabel(TARGET_COLUMN)
    ax2.set_title(f'PC2 vs {TARGET_COLUMN}')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pca_vs_views.png'))
    plt.close()
    print(f"PCA components vs Views visualization saved to {RESULTS_DIR}/pca_vs_views.png")

# ==============================================================================
# Section 7: Split Data into Features (X) and Target (y)
# ==============================================================================
print("\n--- 7: Splitting Data into Features (X) and Target (y) ---")
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in DataFrame.")
    sys.exit(1)

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# ==============================================================================
# Section 8: Split Data into Training and Testing Sets
# ==============================================================================
print("\n--- 8: Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Data split into training and testing sets (test_size={TEST_SIZE}, random_state={RANDOM_STATE}).")

# ==============================================================================
# Section 9: Display Final Shapes
# ==============================================================================
print("\n--- 9: Final Shapes ---")
print("Training features shape (X_train):", X_train.shape)
print("Testing features shape (X_test):", X_test.shape)
print("Training target shape (y_train):", y_train.shape)
print("Testing target shape (y_test):", y_test.shape)
print("\n--- Preprocessing Script Complete ---")

# ==============================================================================
# Section 10: Basic Linear Regression Model
# ==============================================================================
print("\n--- 10: Basic Linear Regression Model ---")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

# Initialize and train the linear regression model
print("Training a basic linear regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on both training and test sets
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = math.sqrt(train_mse)
test_rmse = math.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print("\nLinear Regression Model Performance:")
print(f"Training set MSE: {train_mse:.2f}")
print(f"Test set MSE: {test_mse:.2f}")
print(f"Training set RMSE: {train_rmse:.2f}")
print(f"Test set RMSE: {test_rmse:.2f}")
print(f"Training set MAE: {train_mae:.2f}")
print(f"Test set MAE: {test_mae:.2f}")
print(f"Training set R²: {train_r2:.4f}")
print(f"Test set R²: {test_r2:.4f}")

# Check for feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 most important features in linear regression model:")
print(feature_importance.head(10))

# Plot actual vs predicted values on test set
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Views')
plt.ylabel('Predicted Views')
plt.title('Linear Regression: Actual vs Predicted Views (Test Set)')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'linear_regression_actual_vs_predicted.png'))
plt.close()
print(f"Linear regression actual vs predicted plot saved to {RESULTS_DIR}/linear_regression_actual_vs_predicted.png")

# Plot residuals
plt.figure(figsize=(10, 8))
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Views')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residuals Plot')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'linear_regression_residuals.png'))
plt.close()
print(f"Linear regression residuals plot saved to {RESULTS_DIR}/linear_regression_residuals.png")

# Try a log transformation for better visualization
plt.figure(figsize=(10, 8))
plt.scatter(np.log10(y_test + 1), np.log10(y_test_pred + 1), alpha=0.5)
plt.plot([0, np.log10(y_test.max() + 1)], [0, np.log10(y_test.max() + 1)], 'r--')
plt.xlabel('Log10(Actual Views)')
plt.ylabel('Log10(Predicted Views)')
plt.title('Linear Regression: Log-transformed Actual vs Predicted Views')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'linear_regression_log_transformed.png'))
plt.close()
print(f"Log-transformed actual vs predicted plot saved to {RESULTS_DIR}/linear_regression_log_transformed.png")

print("\n--- Basic Linear Regression Analysis Complete ---")

# ==============================================================================
# Section 11: Converting Regression to Classification for Additional Metrics
# ==============================================================================
print("\n--- 11: Classification Metrics by Binning View Counts ---")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# Define view count bins as mentioned in the project proposal
print("Converting regression problem to classification by binning view counts...")
def categorize_views(views):
    """Categorize views into 4 groups after removing >1B category"""
    if views < 1_000_000:  # Less than 1M
        return 0
    elif views < 10_000_000:  # 1M - 10M
        return 1
    elif views < 100_000_000:  # 10M - 100M
        return 2
    else:  # 100M - 1B (since we removed >1B)
        return 3

# Convert to classification problem by binning
print("\nConverting regression to classification for additional metrics...")
y_train_binned = np.array([categorize_views(v) for v in y_train])  # Use y_train for binning
y_test_binned = np.array([categorize_views(v) for v in y_test])  # Use y_test for binning

# Fit the Random Forest classifier using the binned training data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_classifier.fit(X_train, y_train_binned)

# Predictions and evaluation for classification
y_pred_binned = rf_classifier.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test_binned, y_pred_binned)
precision = precision_score(y_test_binned, y_pred_binned, average='weighted')
recall = recall_score(y_test_binned, y_pred_binned, average='weighted')
f1 = f1_score(y_test_binned, y_pred_binned, average='weighted')

# Print classification metrics
print("\nRandom Forest Classification Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
target_names = ['<1M views', '1M-10M views', '10M-100M views', '100M-1B views']
print(classification_report(y_test_binned, y_pred_binned, target_names=target_names))

# Create confusion matrix
cm = confusion_matrix(y_test_binned, y_pred_binned)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of View Count Categories')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'view_categories_confusion_matrix.png'))
plt.close()
print(f"Confusion matrix saved to {RESULTS_DIR}/view_categories_confusion_matrix.png")

# Display distribution of view categories in test set
plt.figure(figsize=(10, 6))
category_counts = np.bincount(y_test_binned, minlength=4)
category_percentages = category_counts / len(y_test_binned) * 100
bars = plt.bar(target_names, category_counts, color='skyblue')
plt.title('Distribution of View Categories in Test Set')
plt.xlabel('View Category')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Add count and percentage labels on bars
for bar, count, percentage in zip(bars, category_counts, category_percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}\n({percentage:.1f}%)', 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'view_categories_distribution.png'))
plt.close()
print(f"View categories distribution plot saved to {RESULTS_DIR}/view_categories_distribution.png")

print("\n--- Classification Analysis Complete ---")

# ==============================================================================
# Section 12: Random Forest with 5-Fold Cross Validation
# ==============================================================================
print("\n--- 12: Random Forest Model with 5-Fold Cross Validation ---")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Initialize the Random Forest regressor with reasonable parameters
print("Initializing Random Forest regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of trees (None = unlimited)
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1  # Use all available cores
)

# Set up k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
print(f"Performing {k_folds}-fold cross-validation...")

# Regression metrics storage
cv_r2_scores = []
cv_rmse_scores = []
cv_mae_scores = []

# Run k-fold cross-validation for regression metrics
print("\nCross-validation for regression metrics in progress...")
# Get predictions from cross-validation
y_pred_cv = cross_val_predict(rf_model, X, y, cv=kf)

# Calculate R² scores for each fold to get std deviation
cv_r2_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
r2_cv = cv_r2_scores.mean()
r2_cv_std = cv_r2_scores.std()

# Calculate regression metrics once using all predictions
rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
mae_cv = mean_absolute_error(y, y_pred_cv)

# Print regression metrics
print("\nRandom Forest Regression Performance (Cross-validated):")
print(f"Cross-validated RMSE: {rmse_cv:.2f}")
print(f"Cross-validated MAE: {mae_cv:.2f}")
print(f"Cross-validated R²: {r2_cv:.4f} (±{r2_cv_std:.4f})")

# Create a plot for R² scores across folds
plt.figure(figsize=(10, 6))
plt.bar(range(1, k_folds + 1), cv_r2_scores, yerr=r2_cv_std, capsize=10, color='royalblue', alpha=0.7)
plt.axhline(y=r2_cv, color='red', linestyle='--', label=f'Mean R² = {r2_cv:.4f}')
plt.xlabel('Cross-validation Fold')
plt.ylabel('R² Score')
plt.title('Random Forest Regression R² Scores (5-fold CV)')
plt.ylim(max(0, min(cv_r2_scores) - 0.1), min(1.0, max(cv_r2_scores) + 0.1))
plt.xticks(range(1, k_folds + 1))
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_regression_r2_scores.png'))
plt.close()
print(f"R² scores plot saved to {RESULTS_DIR}/rf_regression_r2_scores.png")

# Convert to classification problem by binning
print("\nConverting regression to classification for additional metrics...")
y_true_binned = np.array([categorize_views(v) for v in y])
y_pred_binned_cv = np.array([categorize_views(v) for v in y_pred_cv])

# Calculate classification metrics
accuracy_cv = accuracy_score(y_true_binned, y_pred_binned_cv)
precision_cv = precision_score(y_true_binned, y_pred_binned_cv, average='weighted')
recall_cv = recall_score(y_true_binned, y_pred_binned_cv, average='weighted')
f1_cv = f1_score(y_true_binned, y_pred_binned_cv, average='weighted')

# Print classification metrics
print("\nRandom Forest Classification Performance (Cross-validated):")
print(f"Cross-validated Accuracy: {accuracy_cv:.4f}")
print(f"Cross-validated Precision: {precision_cv:.4f}")
print(f"Cross-validated Recall: {recall_cv:.4f}")
print(f"Cross-validated F1 Score: {f1_cv:.4f}")

# Detailed classification report
print("\nDetailed Classification Report for Random Forest:")
target_names = ['<1M views', '1M-10M views', '10M-100M views', '100M-1B views']
print(classification_report(y_true_binned, y_pred_binned_cv, target_names=target_names))

# Create confusion matrix for Random Forest
cm_rf = confusion_matrix(y_true_binned, y_pred_binned_cv)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest (Cross-validated)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_confusion_matrix.png'))
plt.close()
print(f"Random Forest confusion matrix saved to {RESULTS_DIR}/rf_confusion_matrix.png")

# Train a final model on the entire dataset (for feature importance)
print("\nTraining final Random Forest model on entire dataset for feature importance...")
rf_model.fit(X, y)

# Get feature importances
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 most important features in Random Forest model:")
print(feature_importance_rf.head(10))

# Visualize Random Forest feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf.head(15))
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance.png'))
plt.close()
print(f"Feature importance plot saved to {RESULTS_DIR}/rf_feature_importance.png")

# Compare Linear Regression vs Random Forest
print("\n--- Model Comparison: Linear Regression vs Random Forest ---")
print("Regression Metrics (R²):")
print(f"Linear Regression (Test set): {test_r2:.4f}")
print(f"Random Forest (Cross-validated): {r2_cv:.4f}")
print("\nClassification Accuracy:")
print(f"Linear Regression: {accuracy:.4f}")
print(f"Random Forest: {accuracy_cv:.4f}")

print("\n--- Random Forest Analysis Complete ---")

# ==============================================================================
# Section 13: Random Forest Regression Model
# ==============================================================================
print("\n--- Random Forest Regression Model ---")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
rf_regressor.fit(X_train, y_train)

# Predictions and evaluation for regression
y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

# Calculate regression metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print regression metrics
print("\nRandom Forest Regression Performance:")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# ==============================================================================
# Section 14: Random Forest Classification Model with Cross-Validation
# ==============================================================================
print("\n--- Random Forest Classification Model with Cross-Validation ---")

# Create classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

# Prepare the binned target
y_binned = np.array([categorize_views(v) for v in y])

# Perform cross-validation for classification (k=5 folds)
print(f"Performing {k_folds}-fold cross-validation...")

# Get accuracy, precision, recall, and F1 scores with cross-validation
cv_accuracy_scores = cross_val_score(rf_classifier, X, y_binned, cv=kf, scoring='accuracy')
cv_precision_scores = cross_val_score(rf_classifier, X, y_binned, cv=kf, scoring='precision_weighted')
cv_recall_scores = cross_val_score(rf_classifier, X, y_binned, cv=kf, scoring='recall_weighted')
cv_f1_scores = cross_val_score(rf_classifier, X, y_binned, cv=kf, scoring='f1_weighted')

# Calculate means and standard deviations
accuracy_mean = cv_accuracy_scores.mean()
accuracy_std = cv_accuracy_scores.std()
precision_mean = cv_precision_scores.mean()
precision_std = cv_precision_scores.std()
recall_mean = cv_recall_scores.mean()
recall_std = cv_recall_scores.std()
f1_mean = cv_f1_scores.mean()
f1_std = cv_f1_scores.std()

# Print classification metrics with standard deviations
print("\nRandom Forest Classification Performance (Cross-validated):")
print(f"Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
print(f"Precision: {precision_mean:.4f} (±{precision_std:.4f})")
print(f"Recall: {recall_mean:.4f} (±{recall_std:.4f})")
print(f"F1 Score: {f1_mean:.4f} (±{f1_std:.4f})")

# Create a plot for accuracy scores across folds
plt.figure(figsize=(10, 6))
plt.bar(range(1, k_folds + 1), cv_accuracy_scores, yerr=accuracy_std, capsize=10, color='forestgreen', alpha=0.7)
plt.axhline(y=accuracy_mean, color='red', linestyle='--', label=f'Mean Accuracy = {accuracy_mean:.4f}')
plt.xlabel('Cross-validation Fold')
plt.ylabel('Accuracy')
plt.title('Random Forest Classification Accuracy Scores (5-fold CV)')
plt.ylim(max(0, min(cv_accuracy_scores) - 0.1), min(1.0, max(cv_accuracy_scores) + 0.1))
plt.xticks(range(1, k_folds + 1))
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_classification_accuracy_scores.png'))
plt.close()
print(f"Accuracy scores plot saved to {RESULTS_DIR}/rf_classification_accuracy_scores.png")

# Get predictions for confusion matrix
y_pred_cv = cross_val_predict(rf_classifier, X, y_binned, cv=kf)

# Create confusion matrix
cm = confusion_matrix(y_binned, y_pred_cv)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - RF Classification (Cross-validated)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_classification_confusion_matrix.png'))
plt.close()
print(f"RF Classification confusion matrix saved to {RESULTS_DIR}/rf_classification_confusion_matrix.png")

# Train a final model on all data for feature importance
rf_classifier.fit(X, y_binned)

# Get and visualize feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important features for classification:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Random Forest Classification - Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_classification_feature_importance.png'))
plt.close()
print(f"Classification feature importance plot saved to {RESULTS_DIR}/rf_classification_feature_importance.png")

# Summary comparison
print("\n--- Model Comparison ---")
print("Classification Metrics:")
print(f"Random Forest Regression (converted to classification): {accuracy_cv:.4f}")
print(f"Random Forest Classification (direct): {accuracy_mean:.4f} (±{accuracy_std:.4f})")

# ==============================================================================
# Section 15: Save Metrics for Reporting
# ==============================================================================
print("\n--- Saving Metrics for Reporting ---")

import json

# Create a dictionary with all model metrics
model_metrics = {
    'Linear Regression': {
        'Training R²': train_r2,
        'Test R²': test_r2,
        'Training RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Training MAE': train_mae,
        'Test MAE': test_mae,
        'Classification Accuracy': accuracy
    },
    'Random Forest (Cross-validated)': {
        'Cross-validated R²': r2_cv,
        'Cross-validated R² Std': r2_cv_std,
        'Cross-validated RMSE': rmse_cv,
        'Cross-validated MAE': mae_cv,
        'Cross-validated Accuracy': accuracy_cv,
        'Cross-validated Precision': precision_cv,
        'Cross-validated Recall': recall_cv,
        'Cross-validated F1 Score': f1_cv
    },
    'Random Forest Regression': {
        'Training R²': train_r2,
        'Test R²': test_r2,
        'Training RMSE': train_rmse,
        'Test RMSE': test_rmse
    },
    'Random Forest Classification': {
        'Cross-validated Accuracy': accuracy_mean,
        'Cross-validated Accuracy Std': accuracy_std,
        'Cross-validated Precision': precision_mean,
        'Cross-validated Precision Std': precision_std,
        'Cross-validated Recall': recall_mean,
        'Cross-validated Recall Std': recall_std,
        'Cross-validated F1 Score': f1_mean,
        'Cross-validated F1 Score Std': f1_std
    }
}

# Save metrics to a JSON file in the results directory
metrics_file = os.path.join(RESULTS_DIR, 'model_metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(model_metrics, f, indent=4)

print(f"Model metrics saved to {metrics_file}")
print("\n--- All Analysis Complete ---")

# Suggestion to generate the HTML report
print("\nYou can now generate an HTML report by running:")
print("python generate_report.py")

# Adding a Decision Tree Visualization
print("\n--- Visualizing a Decision Tree ---")

# Create a decision tree classifier with limited depth for visualization
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
dt_classifier.fit(X_train, y_train_binned)

# Create a visualization of the decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, 
          filled=True, 
          feature_names=X.columns.tolist(), 
          class_names=target_names,
          rounded=True,
          fontsize=7)
plt.title('Decision Tree for View Count Classification (Depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'decision_tree_visualization.png'))
plt.close()
print(f"Decision tree visualization saved to {RESULTS_DIR}/decision_tree_visualization.png")

# Similarly, create a regression decision tree
dt_regressor = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
dt_regressor.fit(X_train, y_train)

# Create a visualization of the regression decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_regressor, 
          filled=True, 
          feature_names=X.columns.tolist(), 
          rounded=True,
          fontsize=7)
plt.title('Regression Decision Tree for View Count Prediction (Depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'regression_decision_tree_visualization.png'))
plt.close()
print(f"Regression decision tree visualization saved to {RESULTS_DIR}/regression_decision_tree_visualization.png") 

# Modularize so can use cleaned data in other files
def load_data():
    """
    Returns preprocessed train/test splits and feature names.
    Can be imported and used in other scripts.
    """
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()

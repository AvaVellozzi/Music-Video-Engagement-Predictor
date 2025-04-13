import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import the modularized preprocessing
from preprocessing import load_data

# Load preprocessed data
X_train, X_test, y_train, y_test, feature_names = load_data()

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)
gb_model.fit(X_train, y_train)

# Predict
y_pred = gb_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Gradient Boosting Evaluation ---")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")

# Save visualizations
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Plot: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gb_actual_vs_predicted.png'))
plt.close()

# Plot: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Views")
plt.ylabel("Residuals")
plt.title("Gradient Boosting: Residuals Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gb_residuals.png'))
plt.close()

# Plot: Feature Importances
feature_importance = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.values[:10], y=feature_importance.index[:10])
plt.title("Gradient Boosting: Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gb_feature_importance.png'))
plt.close()

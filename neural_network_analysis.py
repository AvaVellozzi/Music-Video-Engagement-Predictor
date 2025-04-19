import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import preprocessed data
from preprocessing import load_data

# Set results path
RESULTS_DIR = "results/neural_network"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
X_train, X_test, y_train, y_test, feature_names = load_data()

# Normalize target variable for stability (optional)
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

# Define neural network architectures
architectures = {
    "shallow": [64],
    "medium": [128, 64],
    "deep": [256, 128, 64]
}

results = {}

for name, layers in architectures.items():
    print(f"\n--- Training {name.upper()} Neural Network ---")
    model = Sequential()
    model.add(Dense(layers[0], input_dim=X_train.shape[1], activation='relu'))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))  # Regression output

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    print(f"{name.title()} Network - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Views")
    plt.ylabel("Predicted Views")
    plt.title(f"{name.title()} Neural Network: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_actual_vs_predicted.png"))
    plt.close()

# Save results summary
with open(os.path.join(RESULTS_DIR, "nn_summary.txt"), "w") as f:
    for name, metrics in results.items():
        f.write(f"{name.title()} Network:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

print(f"\nNeural network results saved to {RESULTS_DIR}")

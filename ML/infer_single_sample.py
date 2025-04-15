import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the test data
test_df = pd.read_csv("depth_test_with_z.csv")

# Extract features and targets
feature_cols = [col for col in test_df.columns if col.endswith('_x') or col.endswith('_y')]
target_cols = [col for col in test_df.columns if col.endswith('_z')]

X = test_df[feature_cols].to_numpy()
y_true = test_df[target_cols].to_numpy()

# Define window size used during training
window_size = 10
X_windows = np.array([X[i:i + window_size] for i in range(len(X) - window_size + 1)])
y_windows = y_true[window_size - 1:]

# Standardize input
scaler = StandardScaler().fit(X_windows.reshape(-1, X_windows.shape[2]))
X_scaled = scaler.transform(X_windows.reshape(-1, X_windows.shape[2])).reshape(X_windows.shape)

# Load the trained model
model = keras.models.load_model("saved_models/model_fold_1.keras")

# Make predictions
y_pred = model.predict(X_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_windows, y_pred)
print(f"\n🧠 Mean Squared Error: {mse:.6f}\n")

# Print actual vs predicted vs difference
print("🔍 Sample-wise Prediction Comparison:")
for i in range(len(y_pred)):
    actual = y_windows[i]
    predicted = y_pred[i]
    diff = actual - predicted
    print(f"\n🟢 Sample {i+1}:")
    for j, joint in enumerate(target_cols):
        print(f"  {joint:<20} | Actual: {actual[j]: .5f} | Predicted: {predicted[j]: .5f} | Δ: {diff[j]: .5f}")

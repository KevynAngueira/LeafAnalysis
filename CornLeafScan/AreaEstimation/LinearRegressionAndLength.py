# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from GetLeafModelData import GetLeafModelData

num_base_width_segments = 8
skip_segments = 0

precisions = {
    'length': 0,
    'base_widths': 1/16
}

# Load data
X, y = GetLeafModelData(num_base_width_segments, skip_segments, precisions=precisions, pad_factor=4)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train -> MSE: {mse_train:.4f}, R2: {r2_train:.4f}")
print(f"Test  -> MSE: {mse_test:.4f}, R2: {r2_test:.4f}")

# Plot actual vs predicted for test set
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, color='orange', label="Test predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Ideal")
plt.xlabel("Actual Area")
plt.ylabel("Predicted Area")
plt.title(f"Gradient Boosting ({num_base_width_segments} Widths + Length)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"Result/LinearRegression/LinearRegression_{num_base_width_segments}_With_Length.png")

# Save results
results = {
    "model": "Linear Regression",
    "segments": num_base_width_segments,
    "mse_train": mse_train,
    "r2_train": r2_train,
    "mse_test": mse_test,
    "r2_test": r2_test
}
with open(f'Result/LinearRegression/LinearRegression_{num_base_width_segments}_With_Length.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the trained model
joblib.dump(model, "SavedModels/linear_regression_model.pkl")

plt.show()


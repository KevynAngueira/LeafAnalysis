# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-17

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Simulate a small synthetic dataset of leaf widths and original areas
# Each leaf has width measurements at inch 1, 2, 3, ..., 10
# The first three widths are used as predictors (assumed unaffected by defoliation)

np.random.seed(42)

n_samples = 40
n_sections = 10

# Simulate widths with a smooth tapering pattern + some noise
def generate_leaf_profile(base_width):
    taper = np.linspace(1.0, 0.3, n_sections)
    noise = np.random.normal(0, 0.05, n_sections)
    return base_width * taper + noise

leaf_data = np.array([generate_leaf_profile(np.random.uniform(3, 5)) for _ in range(n_samples)])
original_areas = leaf_data.sum(axis=1)  # Simulated "original" leaf area as sum of widths

# Create a DataFrame for easier handling
df = pd.DataFrame(leaf_data, columns=[f"width_{i+1}" for i in range(n_sections)])
df["original_area"] = original_areas

# Baseline 1: Predict original area using the first three widths
X = df[["width_1", "width_2", "width_3"]]
y = df["original_area"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot predictions vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Area")
plt.ylabel("Predicted Area")
plt.title("Baseline Linear Regression (3 Base Widths)")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"MSE: {mse}, R2: {r2}")

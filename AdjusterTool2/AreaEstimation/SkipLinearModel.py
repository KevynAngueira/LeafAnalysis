# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-17

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append("..")
from DefoliationModeller.LeafData import LeafData


leafData = LeafData()

base_width_arr = []
areas_arr = []

num_base_width_segments = 3
skip_segments = 4

for i in range(6, 27):
    leaf_data = leafData.getLeafByID(i)

    # Get Base Widths: The first 3 available widths
    base_widths = list(leaf_data["Start_Width"][skip_segments:num_base_width_segments+skip_segments])
    base_width_arr.append(base_widths)
    print(base_widths)

    # Get Leaf Area: The area column excluding segment 0 (segment 0 not shown in Leaf Scan)
    leaf_segment_areas = leaf_data["Area"]
    leaf_area = leaf_segment_areas[1:].sum()
    areas_arr.append(leaf_area)
    print("Area: ", leaf_area)

# Create a DataFrame for easier handling
df = pd.DataFrame(base_width_arr, columns=[f"width_{i}" for i in range(num_base_width_segments)])
df["original_area"] = areas_arr

print(df)

# Baseline 1: Predict original area using the first three widths
X = df[[f"width_{i}" for i in range(num_base_width_segments)]]
y = df["original_area"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Plot predictions vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Area")
plt.ylabel("Predicted Area")
plt.title(f"Linear Regression ({num_base_width_segments} Base Widths)")
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig(f"Result/LinearRegression/Skip/Linear_Regression_{num_base_width_segments}_Skip_{skip_segments}.png")

# Save results to json
print(f"MSE: {mse}, R2: {r2}")
results = {
    "model": "Linear Regression",
    "segments": num_base_width_segments,
    "skip": skip_segments,
    "mse": mse,
    "r2": r2 
}
with open(f'Result/LinearRegression/Skip/Linear_Regression_{num_base_width_segments}_Skip_{skip_segments}.json', 'w') as f:
    json.dump(results, f, indent=4)

plt.show()

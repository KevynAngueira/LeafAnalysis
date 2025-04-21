# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-21

import sys
import json
import joblib
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
length_arr = []

num_base_width_segments = 3
skip_segments = 4

for i in range(6, 27):
    leaf_data = leafData.getLeafByID(i)

    # Get Base Widths: Skip and select the next segments
    base_widths = list(leaf_data["Start_Width"][skip_segments:num_base_width_segments+skip_segments])
    base_width_arr.append(base_widths)

    # Get Leaf Area: Exclude segment 0
    leaf_segment_areas = leaf_data["Area"]
    leaf_area = leaf_segment_areas[1:].sum()
    areas_arr.append(leaf_area)

    # Calculate effective length (with tapering logic)
    start_widths = list(leaf_data["Start_Width"])
    end_widths = list(leaf_data["End_Width"])
    segment_count = len(start_widths)

    if segment_count < 2:
        effective_length = segment_count
    else:
        last_start = start_widths[-1]
        second_last_end = end_widths[-2]

        if last_start < second_last_end and second_last_end > 0:
            ratio = last_start / second_last_end
            effective_length = (segment_count - 1) + ratio
        else:
            effective_length = segment_count

    length_arr.append(effective_length)

# Create a DataFrame
df = pd.DataFrame(base_width_arr, columns=[f"width_{i}" for i in range(num_base_width_segments)])
df["length"] = length_arr
df["original_area"] = areas_arr

print(df)

# Predict area using base widths + length
X = df[[f"width_{i}" for i in range(num_base_width_segments)] + ["length"]]
y = df["original_area"]

# Train full model
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
plt.title(f"Linear Regression with Length ({num_base_width_segments} Widths after Skipping {skip_segments})")
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig(f"Result/LinearRegression/Skip/Linear_Regression_Length_{num_base_width_segments}_Skip_{skip_segments}.png")

# Save results to json
print(f"MSE: {mse}, R2: {r2}")
results = {
    "model": "Linear Regression + Length",
    "segments": num_base_width_segments,
    "skip": skip_segments,
    "mse": mse,
    "r2": r2
}
with open(f'Result/LinearRegression/Skip/Linear_Regression_Length_{num_base_width_segments}_Skip_{skip_segments}.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the model to a file
joblib.dump(model, "SavedModels/random_forest_model.pkl")

plt.show()

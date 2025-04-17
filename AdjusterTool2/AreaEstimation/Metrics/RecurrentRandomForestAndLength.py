# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-17

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

sys.path.append("..")
from DefoliationModeller.LeafData import LeafData

leafData = LeafData()

# Collect all segment widths into a consistent dataframe
all_widths = []
leaf_lengths = []
max_segments = 14  # Based on your earlier data

for i in range(6, 27):  # Use actual available leaf IDs
    leaf = leafData.getLeafByID(i)
    widths = list(leaf["Start_Width"])
    
    # Store leaf length (non-NaN width values)
    leaf_lengths.append(len(widths))
    
    # Pad with NaNs to ensure consistent length
    if len(widths) < max_segments:
        widths += [np.nan] * (max_segments - len(widths))
    
    all_widths.append(widths)

# Create DataFrame with segment columns
df = pd.DataFrame(all_widths, columns=[f"width_{i}" for i in range(max_segments)])
df["leaf_length"] = leaf_lengths

# Evaluate LOOCV R² for predicting each segment from its previous segments + leaf length
segment_r2s = []

for i in range(1, max_segments):
    prev_cols = [f"width_{j}" for j in range(i)]
    target_col = f"width_{i}"
    
    # Prepare columns including normalized index and leaf length
    temp_df = df[prev_cols + ["leaf_length", target_col]].dropna()
    if temp_df.shape[0] <= 1:
        print(f"Segment {i}: Not enough data for LOOCV")
        segment_r2s.append((i, None))
        continue

    # Add normalized index as a new feature
    temp_df["normalized_index"] = i / temp_df["leaf_length"]

    X = temp_df[prev_cols + ["normalized_index"]].values
    y = temp_df[target_col].values

    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_true.append(y_test[0])
        y_pred.append(model.predict(X_test)[0])
    
    r2 = r2_score(y_true, y_pred)
    segment_r2s.append((i, r2))

# Print results
print("Random Forest + Normalized Index + LOOCV R² for predicting each segment from previous segments:")
for seg, r2 in segment_r2s:
    if r2 is not None:
        print(f"Segment {seg}: R² = {r2:.3f}")
    else:
        print(f"Segment {seg}: Not enough data")

# Plotting results
segments = [seg for seg, r2 in segment_r2s if r2 is not None]
r2_values = [r2 for seg, r2 in segment_r2s if r2 is not None]

plt.figure(figsize=(8, 6))
plt.plot(segments, r2_values, marker='o', linestyle='-', color='green')
plt.title('Random Forest LOOCV R² with Normalized Index Feature')
plt.xlabel('Segment Number')
plt.ylabel('R² Value')
plt.xticks(segments)
plt.grid(True)

# Add a horizontal line at y = 0
plt.axhline(0, color='red', linestyle='-')

# Save the plot to a file
plt.savefig("Metrics/RandomForest_with_length_loocv_r2s_plot.png")

# Show the plot
plt.show()

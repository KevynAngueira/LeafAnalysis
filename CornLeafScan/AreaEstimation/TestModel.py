# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import pandas as pd
import joblib

# --- Config ---
input_file = "/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Summary_Table.xlsx" 
output_file = "Result/RandomForest_Predictions_with_IDs.xlsx"
model_file = "SavedModels/random_forest_model.pkl"

# --- Load model ---
model = joblib.load(model_file)

# --- Load data ---
df = pd.read_excel(input_file, sheet_name="Model")

# Identify width columns dynamically
width_cols = [c for c in df.columns if c.startswith("Width_")]
width_cols = sorted(width_cols, key=lambda x: int(x.split("_")[1]))

# Features (same as training: all widths + length)
X = df[width_cols + ["Length"]]

# Run predictions
y_pred = model.predict(X)

# Build output DataFrame
output_df = df[["F_ID", "P_ID", "L_ID"]].copy()
output_df["Actual_Area"] = df["Original_Area"]
output_df["Predicted_Area"] = y_pred

# Save to Excel
output_df.to_excel(output_file, index=False)

print(f"Predictions saved to {output_file}")

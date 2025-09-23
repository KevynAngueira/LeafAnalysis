# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
input_file = "/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Graph/Original_Area.ods"

# Load ODS file
df = pd.read_excel(input_file, engine="odf")

# Make sure you have a leaf identifier column
# Example: L_ID
if "Leaf_ID" not in df.columns:
    raise ValueError("Column 'L_ID' not found in the dataset!")

# Scatterplot colored by leaf number
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Actual",
    y="Predicted",
    hue="Leaf_ID",        # color code by leaf number
    palette="tab20",   # color palette, adjust if you have >10 leaves
    s=70,              # marker size
    alpha=0.8
)

# Add diagonal line for perfect predictions
plt.plot([df["Actual"].min(), df["Actual"].max()],
         [df["Actual"].min(), df["Actual"].max()],
         "k--", lw=2, label="Ideal (y=x)")

plt.xlabel("True Area (Integrated)")
plt.ylabel("LeafScan Area")
plt.title("Original Area: LeafScan vs True")
plt.legend(title="Leaf Number", bbox_to_anchor=(1.05,1), loc="upper left")
plt.grid(True)
plt.tight_layout()

output_file = "Original_Area_Scatter.png"
plt.savefig(output_file, dpi=300)
plt.show()

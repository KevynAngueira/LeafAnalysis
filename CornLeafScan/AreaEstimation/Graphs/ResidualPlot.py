# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
input_file = "/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Graph/Original_Area.ods"

# Load ODS
df = pd.read_excel(input_file, engine="odf")

# Compute residuals
df["Residual"] = df["Predicted"] - df["Actual"] 

# Residual plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Actual",
    y="Residual",
    hue="Leaf_ID",
    palette="tab20",   # choose a color palette
    s=70,
    alpha=0.8
)

plt.axhline(0, color="k", linestyle="--", lw=2)
plt.xlabel("True Area (Integrated)")
plt.ylabel("LeafScan Error")
plt.title("Original Area: LeafScan vs True")
plt.legend(title="Leaf Number", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

output_file = "Original_Area_Residual.png"
plt.savefig(output_file, dpi=300)
plt.show()

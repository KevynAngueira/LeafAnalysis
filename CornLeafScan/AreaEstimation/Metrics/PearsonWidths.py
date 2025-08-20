# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append("..")
from DefoliationModeller.LeafData import LeafData

leafData = LeafData()

max_segments = 15  # Adjust if needed
segment_widths_by_leaf = []
leaf_areas = []

# Collect all widths and areas
for i in range(6, 27):
    leaf = leafData.getLeafByID(i)
    widths = leaf["Start_Width"]
    area = leaf["Area"][1:].sum()  # Exclude segment 0

    # Pad widths with NaN to align across all leaves
    padded_widths = np.full(max_segments, np.nan)
    padded_widths[:len(widths)] = widths
    segment_widths_by_leaf.append(padded_widths)
    leaf_areas.append(area)

# Convert to DataFrame
width_df = pd.DataFrame(segment_widths_by_leaf, columns=[f"width_{i}" for i in range(max_segments)])
width_df["original_area"] = leaf_areas

# Compute Pearson correlation for each segment
correlations = []
for i in range(max_segments):
    seg_col = f"width_{i}"
    if width_df[seg_col].isna().all():
        continue
    corr, _ = pearsonr(width_df[seg_col].dropna(), width_df["original_area"].loc[width_df[seg_col].notna()])
    correlations.append((seg_col, corr))

# Sort by segment index (already in order, but being explicit)
correlations.sort(key=lambda x: int(x[0].split("_")[1]))

# Print results
print("Segment Correlations with Original Area (ordered by segment index):")
for seg, corr in correlations:
    print(f"{seg}: {corr:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.bar([seg for seg, _ in correlations], [corr for _, corr in correlations], color='steelblue')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45)
plt.ylabel("Pearson Correlation with Area")
plt.title("Segment Width vs. Leaf Area Correlation (by Segment Index)")
plt.tight_layout()
plt.savefig("Segment_Correlation_With_Area.png")
plt.show()

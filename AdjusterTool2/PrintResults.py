# Author: Kevyn Angueira Irizarry
# Created: 2025-04-21
# Last Modified: 2025-04-21

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_FILE = "batch_defoliation_results.csv"
OUTLIERS_FILE = "batch_outliers.txt"

def print_mae(title, data):
    print(f"{title}")
    print(f"🟩 MAE - Original Area: {np.mean(np.abs(data['estimated_original_area'] - data['real_original_area'])):.2f}")
    print(f"🟨 MAE - Remaining Area: {np.mean(np.abs(data['calculated_remaining_area'] - data['real_remaining_area'])):.2f}")
    print(f"🟥 MAE - Defoliation %: {np.mean(np.abs(data['estimated_defoliation'] - data['real_defoliation'])):.2f}
")

def main():
    df = pd.read_csv(RESULTS_FILE)
    print("
📊 --- Final Summary ---
")

    print_mae("📁 ALL DATA", df)

    outliers_path = Path(OUTLIERS_FILE)
    if outliers_path.exists():
        with open(outliers_path) as f:
            # Extract just the filenames from the full paths
            outlier_filenames = set(Path(line.strip()).name for line in f if line.strip())

        df_filtered = df[~df.apply(
            lambda row: f"leaf-{int(row['leaf_id']):03d}_defoliated-{int(row['real_defoliation'])}_vid-{int(row['media_id']):02d}.mp4" in outlier_filenames,
            axis=1
        )]

        print_mae("📁 WITHOUT OUTLIERS", df_filtered)
        print(f"🚫 Outliers recorded: {len(outlier_filenames)} → loaded from {OUTLIERS_FILE}")
    else:
        print("🚫 No outlier file found.")

if __name__ == "__main__":
    main()

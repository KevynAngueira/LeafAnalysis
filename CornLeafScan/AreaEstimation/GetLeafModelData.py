# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-09-22

import sys
import numpy as np
import pandas as pd

def GetLeafModelData(
    num_base_width_segments,
    skip_segments,
    precisions=None,
    pad_factor=1,
    include_length=True
):
    """
    Load leaf model data from an Excel file instead of LeafData.

    Parameters
    ----------
    num_base_width_segments : int
        Number of width columns to extract (after skipping).
    skip_segments : int
        How many initial width columns to skip before counting.
    precisions : dict, optional
        Dict specifying measurement noise levels, e.g. {"base_widths": 0.1, "length": 0.2}.
    pad_factor : int, optional
        Number of noisy replicates to generate (including the original). Default = 1.
    include_length : bool, optional
        Whether to include the `length` column as a feature.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (original area).
    """

    # Load file
    df = pd.read_excel("/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Summary_Table.xlsx", sheet_name="Model")

    # Identify available width columns dynamically
    width_cols = [c for c in df.columns if c.startswith("Width_")]
    width_cols = sorted(width_cols, key=lambda x: int(x.split("_")[1]))  # Ensure order

    # Subset width columns according to skip + num_base_width_segments
    selected_width_cols = width_cols[skip_segments:skip_segments + num_base_width_segments]

    # Extract arrays
    base_width_arr = df[selected_width_cols].values.tolist()
    length_arr = df["Length"].values.tolist()
    areas_arr = df["Original_Area"].values.tolist()

    # Apply padding (synthetic noisy copies)
    if precisions and pad_factor > 1:
        width_noise = precisions.get("base_widths", 0)
        length_noise = precisions.get("length", 0)

        for _ in range(pad_factor - 1):
            noisy_widths = [
                [w + np.random.normal(-width_noise, width_noise) for w in row]
                for row in base_width_arr
            ]
            noisy_length = [l + np.random.normal(-length_noise, length_noise) for l in length_arr]

            base_width_arr.extend(noisy_widths)
            length_arr.extend(noisy_length)
            areas_arr.extend(areas_arr)  # duplicate ground-truth areas

    # Rebuild into dataframe
    out_df = pd.DataFrame(base_width_arr, columns=[f"Width_{i}" for i in range(num_base_width_segments)])
    out_df["Length"] = length_arr
    out_df["Original_Area"] = areas_arr

    # Feature/target selection
    if include_length:
        X = out_df[[f"Width_{i}" for i in range(num_base_width_segments)] + ["Length"]]
    else:
        X = out_df[[f"Width_{i}" for i in range(num_base_width_segments)]]
    y = out_df["Original_Area"]

    return X, y

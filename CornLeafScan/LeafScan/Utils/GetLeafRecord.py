# Author: Kevyn Angueira Irizarry
# Created: 2025-09-24
# Last Modified: 2025-09-24

import pandas as pd

def GetLeafRecord(f_id, p_id, l_id, skip_segments=0, num_widths= 8, summary_path="/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Summary_Table.xlsx"):
    """
    Retrieve leaf metadata and measurements for a given field, plant, and leaf ID.

    Parameters
    ----------
    f_id : str or int
        Field ID to query.
    p_id : str or int
        Plant ID to query.
    l_id : str or int
        Leaf ID to query.
    summary_path : str
        Path to the Excel summary table.

    Returns
    -------
    record : dict or None
        A dictionary containing the leaf's metadata and measurements, or None if not found.
    """

    # Load summary sheet
    df = pd.read_excel(summary_path, sheet_name="Model")

    # Normalize IDs for matching (string compare)
    f_id = str(f_id).zfill(3) if isinstance(f_id, int) or f_id.isdigit() else str(f_id)
    p_id = str(p_id).zfill(3) if isinstance(p_id, int) or p_id.isdigit() else str(p_id)
    l_id = str(l_id).zfill(3) if isinstance(l_id, int) or l_id.isdigit() else str(l_id)

    # Locate the matching row
    match = df[
        (df["F_ID"].astype(str).str.zfill(3) == f_id) &
        (df["P_ID"].astype(str).str.zfill(3) == p_id) &
        (df["L_ID"].astype(str).str.zfill(3) == l_id)
    ]

    if match.empty:
        return None, None

    row = match.iloc[0]

    # Build width column names dynamically
    width_cols = [c for c in df.columns if c.startswith("Width_")]
    width_cols = sorted(width_cols, key=lambda x: int(x.split("_")[1]))
    selected_widths = width_cols[skip_segments: skip_segments + num_widths]

    # Build X and y
    X = row[selected_widths + ["Length"]].to_frame().T  # keep DataFrame shape
    y = pd.Series([row["Original_Area"]])

    return X, y
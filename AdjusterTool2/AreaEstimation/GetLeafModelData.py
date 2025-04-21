# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-21

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append("..")
from DefoliationModeller.LeafData import LeafData


def GetLeafModelData(num_base_width_segments, skip_segments, include_length=True):
    leafData = LeafData()

    base_width_arr = []
    areas_arr = []
    length_arr = []

    for i in range(6, 27):
        leaf_data = leafData.getLeafByID(i)

        # Use the first `num_base_width_segments`
        base_widths = list(leaf_data["Start_Width"][skip_segments:skip_segments+num_base_width_segments])
        base_width_arr.append(base_widths)

        # Area: sum of all segments except 0
        leaf_segment_areas = leaf_data["Area"]
        leaf_area = leaf_segment_areas[1:].sum()
        areas_arr.append(leaf_area)

        # Length with taper adjustment
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

    # DataFrame setup
    df = pd.DataFrame(base_width_arr, columns=[f"width_{i}" for i in range(num_base_width_segments)])
    df["length"] = length_arr
    df["original_area"] = areas_arr

    print(df)

    if include_length:
        X = df[[f"width_{i}" for i in range(num_base_width_segments)] + ["length"]]
    else:
        X = df[[f"width_{i}" for i in range(num_base_width_segments)]]
    y = df["original_area"]

    return X, y

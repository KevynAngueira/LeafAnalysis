# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append("..")
from DefoliationModeller.LeafData import LeafData

def GetLeafModelData(num_base_width_segments, skip_segments, precisions=None, include_length=True, pad_factor=1):
    leafData = LeafData()

    base_width_arr = []
    areas_arr = []
    length_arr = []

    for i in range(6, 27):
        base_widths = list(leafData.getWidthsByID(i))[skip_segments:skip_segments+num_base_width_segments]
        original_length, _ = leafData.getLengthsByID(i)
        area = leafData.getAreaByID(i)

        # Always add original data
        base_width_arr.append(base_widths)
        length_arr.append(original_length)
        areas_arr.append(area)

        # Add padded versions
        if precisions and pad_factor > 1:
            for _ in range(pad_factor - 1):
                width_noise = precisions.get("base_widths", 0)
                length_noise = precisions.get("length", 0)

                noisy_widths = [w + np.random.normal(-width_noise, width_noise) for w in base_widths]
                noisy_length = original_length + np.random.normal(-length_noise, length_noise)
                base_width_arr.append(noisy_widths)
                length_arr.append(noisy_length)
                areas_arr.append(area)  # Area stays the same since the noise is synthetic

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


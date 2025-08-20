# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import cv2
import numpy as np

class LeafWidthExtractor:
    def __init__(self, max_segments=3, pixel_to_inch_scale=0.01, row_padding=5, row_window=5):
        """
        max_segments: number of widths to collect
        pixel_to_inch_scale: scaling factor from pixels to inches
        row_padding: number of rows to skip from top and bottom to avoid black padding
        row_window: number of rows to average width over after padding
        """
        self.pixel_to_inch_scale = pixel_to_inch_scale  # 650 px = 6.5 inches
        self.widths = []
        self.max_segments = max_segments
        self.row_padding = row_padding
        self.row_window = row_window

    def resetWidths(self):
        self.widths = []

    def extractWidth(self, leaf_mask):
        if len(self.widths) >= self.max_segments:
            return None  # Stop after collecting desired number of widths

        # Convert to binary if not already
        if leaf_mask.max() > 1:
            leaf_mask = (leaf_mask == 255).astype(np.uint8)

        h, w = leaf_mask.shape
        top_rows = leaf_mask[self.row_padding : self.row_padding + self.row_window, :]
        bottom_rows = leaf_mask[h - self.row_padding - self.row_window : h - self.row_padding, :]

        top_width_avg = np.count_nonzero(top_rows, axis=1).mean()
        bottom_width_avg = np.count_nonzero(bottom_rows, axis=1).mean()

        chosen_width_px = min(top_width_avg, bottom_width_avg)
        chosen_width_in = chosen_width_px * self.pixel_to_inch_scale

        self.widths.append(chosen_width_in)
        return chosen_width_in

    def getWidths(self):
        return self.widths

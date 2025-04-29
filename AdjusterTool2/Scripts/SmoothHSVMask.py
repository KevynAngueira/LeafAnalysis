# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-04-29


import cv2
import numpy as np

from Scripts.HSVMask import HSVMask 

class SmoothHSVMask(HSVMask):
    def __init__(self, hsv_bounds, sat_threshold=None, low_sat_bounds=None, alpha=0.2, binary_threshold=200):
        super().__init__(hsv_bounds, sat_threshold, low_sat_bounds)

        self.prev_mask = None

        self.alpha=alpha
        self.binary_threshold = binary_threshold

    def __smooth_mask(self, prev_mask, new_mask):
        """Apply EMA for mask smoothing."""
        if prev_mask is None:
            return new_mask
        return cv2.addWeighted(prev_mask, 1 - self.alpha, new_mask, self.alpha, 0)

    def applyHSVMask(self, image, invert_range=False, preprocess=True, stabilize=True):
        """
        Applies one of the two HSV masks dynamically based on the background type.
            Colored In Background -> Base Mask
            Black/White Background -> Low Saturation Mask
        """
        result, dynamic_mask = super().applyHSVMask(image, invert_range, preprocess)

        smoothed_mask = self.__smooth_mask(self.prev_mask, dynamic_mask)
        self.prev_mask = smoothed_mask

        _, binary_mask = cv2.threshold(smoothed_mask, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # Overlay on original image
        result = cv2.bitwise_and(image, image, mask=binary_mask)

        return result, dynamic_mask
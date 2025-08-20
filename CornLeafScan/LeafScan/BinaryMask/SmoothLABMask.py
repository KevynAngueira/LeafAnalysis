# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20


import cv2
import numpy as np
 
from .LABMask import LABMask 

class SmoothLABMask(LABMask):
    def __init__(self, hsv_bounds, alpha=0.2, binary_threshold=200):
        super().__init__(hsv_bounds)

        self.prev_mask = None

        self.alpha=alpha
        self.binary_threshold = binary_threshold

    def reset(self):
        self.prev_mask = None

    def __smooth_mask(self, prev_mask, new_mask):
        """Apply EMA for mask smoothing."""
        if prev_mask is None:
            return new_mask
        return cv2.addWeighted(prev_mask, 1 - self.alpha, new_mask, self.alpha, 0)

    def applyMask(self, image, stabilize=True, invert_range=False, preprocess=True):
        """
        Applies one of the two HSV masks dynamically based on the background type.
            Colored In Background -> Base Mask
            Black/White Background -> Low Saturation Mask
        """
        result, dynamic_mask = super().applyMask(image, invert_range, preprocess)

        if stabilize:
            smoothed_mask = self.__smooth_mask(self.prev_mask, dynamic_mask)
            self.prev_mask = smoothed_mask

            _, binary_mask = cv2.threshold(smoothed_mask, self.binary_threshold, 255, cv2.THRESH_BINARY)

            # Overlay on original image
            result = cv2.bitwise_and(image, image, mask=binary_mask)

        return result, dynamic_mask
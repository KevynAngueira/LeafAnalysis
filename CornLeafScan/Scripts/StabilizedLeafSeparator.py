# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20


import cv2
import numpy as np
from dataclasses import dataclass

from Scripts.SmoothHSVMask import SmoothHSVMask
from Scripts.SmoothLABMask import SmoothLABMask
from Scripts.LABMask import LABMask

from Scripts.ResizeForDisplay import resize_for_display
from Scripts.LeafSeparator import LeafSeparator, LeafSeparatorConfig

class StabilizedLeafSeparator(LeafSeparator):
    def __init__(self, config: LeafSeparatorConfig=None, alpha_cropping=0.1, alpha_mask=0.75):
        super().__init__(config)

        # Initialize smoothing variables for cropping
        self.prev_left_bound = None
        self.prev_right_bound = None
        self.prev_top_bound = None
        self.prev_bottom_bound = None
        self.alpha_cropping = alpha_cropping

        #self.leafMask = SmoothLABMask(config.leaf_bounds, alpha=alpha_mask)
        self.leafMask = LABMask(config.leaf_bounds)
    
    def resetLeafSeparator(self):
        self.prev_left_bound = None
        self.prev_right_bound = None
        self.prev_top_bound = None
        self.prev_bottom_bound = None
        #self.leafMask.resetMask()
    
    def __smooth_bound(self, prev_value, new_value):
        """Apply exponential moving average (EMA) for smooth transitions."""
        if prev_value is None:
            return new_value
        return self.alpha_cropping * new_value + (1 - self.alpha_cropping) * prev_value

    def _crop_using_contours(self, image, stabilize=True):
        """Stabilized cropping using exponential moving average for boundaries."""
        cropped_image, bounds = super()._crop_using_contours(image)

        if stabilize:
            top_bound, bottom_bound, left_bound, right_bound = bounds

            # Compute average values, using smoothing
            left_bound = self.__smooth_bound(self.prev_left_bound, left_bound)
            right_bound = self.__smooth_bound(self.prev_right_bound, right_bound)
            top_bound = self.__smooth_bound(self.prev_top_bound, top_bound)
            bottom_bound = self.__smooth_bound(self.prev_bottom_bound, bottom_bound)

            # Store previous values
            self.prev_left_bound, self.prev_right_bound = left_bound, right_bound
            self.prev_top_bound, self.prev_bottom_bound = top_bound, bottom_bound

            # Crop the image
            cropped_image = image[int(top_bound):int(bottom_bound), int(left_bound):int(right_bound)]
        
        return cropped_image, bounds
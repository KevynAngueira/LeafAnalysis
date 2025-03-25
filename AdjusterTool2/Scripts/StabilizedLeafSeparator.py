# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-03-25


import cv2
import numpy as np
from dataclasses import dataclass

from Scripts.SmoothHSVMask import SmoothHSVMask
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

        self.leafMask = SmoothHSVMask(config.leaf_bounds, config.sat_threshold, config.low_sat_leaf_bounds, alpha=alpha_mask)
    
    def __smooth_bound(self, prev_value, new_value):
        """Apply exponential moving average (EMA) for smooth transitions."""
        if prev_value is None:
            return new_value
        return self.alpha_cropping * new_value + (1 - self.alpha_cropping) * prev_value

    def _crop_using_contours(self, image):
        """Stabilized cropping using exponential moving average for boundaries."""
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        left_points, right_points, top_points, bottom_points = [], [], [], []
        for contour in contours:
            for point in contour[:, 0]:  # Extract (x, y) points
                x, y = point
                if x <= self.border_margin:
                    left_points.append(x)
                if x >= width - self.border_margin:
                    right_points.append(x)
                if y <= self.border_margin:
                    top_points.append(y)
                if y >= height - self.border_margin:
                    bottom_points.append(y)

        # Compute average values, using smoothing
        left_bound = self.__smooth_bound(self.prev_left_bound, int(np.mean(left_points))) if left_points else 0
        right_bound = self.__smooth_bound(self.prev_right_bound, int(np.mean(right_points))) if right_points else width
        top_bound = self.__smooth_bound(self.prev_top_bound, int(np.mean(top_points))) if top_points else 0
        bottom_bound = self.__smooth_bound(self.prev_bottom_bound, int(np.mean(bottom_points))) if bottom_points else height

        # Store previous values
        self.prev_left_bound, self.prev_right_bound = left_bound, right_bound
        self.prev_top_bound, self.prev_bottom_bound = top_bound, bottom_bound

        # Crop the image
        cropped_image = image[int(top_bound):int(bottom_bound), int(left_bound):int(right_bound)]
        return cropped_image
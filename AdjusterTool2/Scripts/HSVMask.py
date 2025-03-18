# Author: Your Name
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:43:17


import cv2
import numpy as np

from Scripts.NormalizeBrightness import normalizeBrightness

class HSVMask:
    def __init__(self, hsv_bounds, sat_threshold=None, low_sat_bounds=None):
        self.hsv_bounds = hsv_bounds
        
        if sat_threshold is None:
            self.sat_threshold = 0
        else:
            self.sat_threshold = sat_threshold

        if low_sat_bounds is None:
            self.low_sat_bounds = (np.array([0, self.sat_threshold, 0]), np.array([179, 255, 255]))
        else:
            self.low_sat_bounds = low_sat_bounds

    def __imagePreprocessing(self, image):
        """
        Applying preprocessing to image
            HSV -> Convert to HSV
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        return hsv
        
    def __isColorBackground(self, hsv):
        """
        Determine whether the background is predominantly colored in or black/white
        Uses the mean saturation value of the image as a heuristic
        """
    
        mean_saturation = np.mean(hsv[:, :, 1])
        is_color_background = mean_saturation > self.sat_threshold
    
        return is_color_background
    
    def applyHSVMask(self, image, invert_range=False, preprocess=True):
        """
        Applies one of the two HSV masks dynamically based on the background type.
            Colored In Background -> Hue Mask
            Black/White Background -> Saturation Mask
        """
        
        # Apply preprocessing
        if preprocess:
            hsv = self.__imagePreprocessing(image)
        else:
            hsv = image

        # Check which mask to use
        is_color_background = self.__isColorBackground(hsv)

        if is_color_background:
            # Apply Hue Mask
            lower_bound, upper_bound = self.hsv_bounds
        else:
            # Apply Saturation Mask
            lower_bound, upper_bound = self.low_sat_bounds
        
        # Apply Mask, check for if inverted range
        dynamic_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        if invert_range:
            dynamic_mask = cv2.bitwise_not(dynamic_mask)

        # Overlay on original image
        result = cv2.bitwise_and(image, image, mask=dynamic_mask)

        return result, dynamic_mask
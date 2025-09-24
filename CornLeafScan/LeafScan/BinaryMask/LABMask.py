# Author: Kevyn Angueira Irizarry
# Created: 2025-03-25
# Last Modified: 2025-09-24


import cv2
import numpy as np

class LABMask:
    def __init__(self, lab_bounds):
        self.lab_bounds = lab_bounds
    
    def _imagePreprocessing(self, image):
        """
        Applying preprocessing to image
            CIE LAB -> Convert from BGR to CIE LAB
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        return lab
    
    def applyMask(self, image, invert_range=False, preprocess=True, **kwargs):
        """
        Applies CIE LAB mask based on bounds
        """        
        
        # Return empty mask if image is empty or completely black
        if image is None or image.size == 0 or not np.any(image):
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            empty_result = np.zeros_like(image)
            return empty_result, empty_mask

        # Apply preprocessing
        if preprocess:
            lab = self._imagePreprocessing(image)
        else:
            lab = image

        lower_bound, upper_bound = self.lab_bounds

        # Apply Mask, check for if inverted range
        lab_mask = cv2.inRange(lab, lower_bound, upper_bound)
        if invert_range:
            lab_mask = cv2.bitwise_not(lab_mask)

        # Overlay on original image
        result = cv2.bitwise_and(image, image, mask=lab_mask)

        return result, lab_mask
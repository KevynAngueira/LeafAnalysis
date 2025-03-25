# Author: Kevyn Angueira Irizarry
# Created: 2025-03-25
# Last Modified: 2025-03-25


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
        # Convert to HSV
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        return lab
    
    def applyMask(self, image, invert_range=False, preprocess=True):
        """
        Applies CIE LAB mask based on bounds
        """        
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
# Author: Kevyn Angueira Irizarry
# Created: 2025-02-07
# Last Modified: 2025-03-18


import cv2
import numpy as np

def resize_for_display(image, max_width=1000, max_height=800):
    """ Resize image while maintaining aspect ratio for display. """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))
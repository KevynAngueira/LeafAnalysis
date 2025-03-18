# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-03-18


import cv2
import numpy as np

def normalizeBrightness(hsv_image):
    """ Normalize the brightness by equalizing the histogram on the value channel. Normalizes shadows and glare. """
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])  # Normalize brightness
    return hsv_image
import cv2
import numpy as np


def normalizeBrightness(hsv_image):
    """ Normalize the brightness by equalizing the histogram on the value channel. Normalizes shadows and glare. """
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])  # Normalize brightness
    return hsv_image
    
def dynamicValueThreshold(hsv_image):
    """ Returns a dynamic value channel threshold based on average brightness """
    avg_brightness = np.mean(hsv_image[:, :, 2])

    # Adjust lower V dynamically (darker scenes need a lower threshold)
    lower_v = max(100, int(avg_brightness))
    return lower_v
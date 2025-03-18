# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:45:13


import cv2
import numpy as np

def cropAndRotate(image, min_rect):
    """ 
    Rotate image around the rectangle's center and then crop to dimensions.
    """

    if min_rect is None:
        print("Could not crop and rotate image, invalid rectangle")
        return image

    center, (w, h), angle = min_rect

    # Ensure width is always the greater measurement (left-to-right)
    if w < h:
        w, h = h, w
        angle += 90  # Rotate to correct orientation

    # Rotate the entire image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Extract the correctly rotated bounding box
    x, y, w, h = int(center[0] - w / 2), int(center[1] - h / 2), int(w), int(h)

    # Clamp cropping coordinates to be inside image dimensions
    height, width = rotated.shape[:2] 
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)

    cropped = rotated[y:y+h, x:x+w]

    return cropped

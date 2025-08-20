# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import cv2
import numpy as np

def cropAndRotate(image, min_rect, is_width_greater=True):
    """ 
    Rotate image around the rectangle's center and then crop to dimensions.
    """

    if min_rect is None:
        print("Could not crop and rotate image, invalid rectangle")
        return image

    center, (w, h), angle = min_rect

    # Ensure width is always the greater measurement (left-to-right)
    #if is_width_greater and w < h:
    #    w, h = h, w
    #    angle += 90 
    
    #if not is_width_greater and h < w:
    #    w, h = h, w
    #    angle -= 90

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute bounding box of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    h_img, w_img = image.shape[:2]

    new_w = int((h_img * sin) + (w_img * cos))
    new_h = int((h_img * cos) + (w_img * sin))

    # Adjust rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate with expanded canvas
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    # Adjust center point since image dimensions changed
    center = (new_w / 2, new_h / 2)

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

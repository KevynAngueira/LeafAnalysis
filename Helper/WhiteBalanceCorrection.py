import cv2
import numpy as np

def resize_for_display(image, max_width=1000, max_height=800):
    """ Resize image while maintaining aspect ratio for display. """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

image = cv2.imread("leaf_image2.jpg")
balanced_image = white_balance(image)

cv2.imshow("Original", resize_for_display(image))
cv2.imshow("White Balanced", resize_for_display(balanced_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

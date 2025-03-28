import cv2
import numpy as np
from CropAndRotate import cropAndRotate
from DetectRectangle import detectRectangle
from ExtractGreen import extractGreen
from Helper.ResizeForDisplay import resize_for_display

# Load image
img_path = "TestImages/leaf_image1.jpg"

image = cv2.imread(img_path)

# Define tool color adaptive mask
lower_orange = np.array([2, 115, -1]) # Tool lower value
upper_orange = np.array([12, 255, 255]) # Tool upper value
orange_hsv_thresholds = (lower_orange, upper_orange)

lower_leaf = np.array([35, 23, 0]) # Tool lower value
upper_leaf = np.array([100, 255, 215]) # Tool upper value
leaf_hsv_thresholds = (lower_leaf, upper_leaf)

# Define target dimensions
rectangle_dimensions = (6.5, 1)

# Detect the target rectangle
target_box, target_rect = detectRectangle(image, orange_hsv_thresholds, rectangle_dimensions)

# Draw the largest detected target box in green
if target_box is not None:
    cv2.drawContours(image, [target_box], 0, (0, 255, 0), 3) 
    cv2.imshow("Detected Contours", resize_for_display(image))
    
    image = cv2.imread(img_path)
    cropped_image = cropAndRotate(image, target_rect)
    cv2.imshow("Cropped & Rotated", resize_for_display(cropped_image))
    cv2.imwrite("cropped_output.jpg", cropped_image)  # Save the cropped image

    result, leaf_area = extractGreen(cropped_image, orange_hsv_thresholds, leaf_hsv_thresholds, rectangle_dimensions)
    
else: 
    print('Target rectangle not found')
    cv2.imshow("Detected Contours", resize_for_display(image))

cv2.waitKey(0)
cv2.destroyAllWindows()
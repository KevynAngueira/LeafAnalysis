import cv2
import numpy as np

from Helper.ResizeForDisplay import resize_for_display
from DetectRectangle import detectRectangle

image_path = "/home/icicle/VSCode/LeafAnalysis/ColorTests/NewTool/NewTool.jpg"
#image_path = "/home/icicle/VSCode/LeafAnalysis/TestImages/leaf_image1.jpg"

image = cv2.imread(image_path)

#pink_hsv_thresholds = (np.array([2, 115, 0]), np.array([12, 255, 255]))
pink_hsv_thresholds = (np.array([130, 0, 0]), np.array([179, 255, 255]))
target_dimensions = (6.5, 1)

target_box, target_rect = detectRectangle(image, pink_hsv_thresholds, target_dimensions, draw=True)

cv2.imshow("Rectangle", resize_for_display(image))
cv2.waitKey(0)
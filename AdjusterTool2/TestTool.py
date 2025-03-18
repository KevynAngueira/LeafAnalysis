# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:45:13


import os
import cv2
import numpy as np

from Scripts.HSVMask import HSVMask
from Scripts.ViewWindow import ViewWindow
from Scripts.CropAndRotate import cropAndRotate
from Scripts.ResizeForDisplay import resize_for_display

# Load images from folder
base_folder = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/Results" 
image_files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f))]

if image_files is None:
    print("No Images Found")

print("Select an image by entering the corresponding number:")
for i, filename in enumerate(image_files):
    print(f"{i}: {filename}")

while True:
    try:
        choice = int(input("Enter your choice: "))
        if 0 <= choice < len(image_files):
            filename = image_files[choice]
            image_path = os.path.join(base_folder, filename)
            break
        else:
            print("Invalid selection. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    viewWindow = ViewWindow()
    view_window = viewWindow.Extract(image, True)


    """
    # Tool hue mask
    tool_hsv_bounds = (np.array([4, 0, 0]), np.array([172, 255, 255]))

    # Tool target dimensions
    target_dimensions = (6.5, 1)

    # Extract Tool 
    tool_mask = HSVMask(tool_hsv_bounds)
    tool_hsv, mask = tool_mask.applyHSVMask(image)
    
    # Detect View Window
    preprocessed = ViewWindow.imagePreprocessing(mask)
    contours = ViewWindow.getContours(preprocessed)

    drawn_contours = ViewWindow.drawContours(image, contours, None)

    target_box, target_rect = ViewWindow.ViewWindowFromContours(contours)

    print("==== Target Box ====")
    print(target_box)
    print("==== Target Rect ====")
    print(target_rect)

    target_countour = ViewWindow.drawContours(image, [target_box], (0, 255, 255))
    result = cropAndRotate(image, target_rect)

    cv2.imshow("Original Image", resize_for_display(image))
    cv2.imshow("Mask", resize_for_display(mask))
    cv2.imshow("Tool HSV", resize_for_display(tool_hsv))
    cv2.imshow("Drawn Contours", resize_for_display(drawn_contours))
    cv2.imshow("Target Contour", resize_for_display(target_countour))
    cv2.imshow("Result", resize_for_display(result))

    save_path = os.path.join(base_folder, 'Results', filename)
    cv2.imwrite(save_path, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
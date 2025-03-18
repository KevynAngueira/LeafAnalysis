# Author: Your Name
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:43:18


import os
import cv2
import numpy as np

from Scripts.LeafSeparator import LeafSeparator
from Scripts.ResizeForDisplay import resize_for_display


# Load images from folder
base_folder = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Images/Leaf/Results"
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

leafSeparator = LeafSeparator()

leaf_result, leaf_pixels = leafSeparator.Extract(image, display=True)
 

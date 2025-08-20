# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import cv2
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from colour.characterisation.datasets import CCS_COLOURCHECKERS
from PIL import Image

# === Step 1: Load Reference Image ===
image = cv2.imread("Images/Reference.jpg")
if image is None:
    raise FileNotFoundError("Reference image not found.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Step 2: Detect ColorCheckers ===
checkers = detect_colour_checkers_segmentation(image_rgb)
if not checkers:
    raise ValueError("No ColorChecker detected in the image!")
checker = checkers[0]  # use first detected

# === Step 3: Extract 24 Patch Colors ===
patch_data = np.array(checker.data)
patches = patch_data.reshape(24, -1, 3)  # shape: (24, pixels_per_patch, 3)
detected_colors = np.array([np.mean(patch, axis=0) for patch in patches])

# === Step 4: Create 6x4 Grid Image of Patch Colors ===
def create_color_grid(patch_colors, rows=4, cols=6, patch_size=100):
    # Convert from [0,1] to [0,255] before saving as uint8
    patch_colors = np.clip(patch_colors * 255, 0, 255).astype(np.uint8)
    grid_img = np.zeros((rows * patch_size, cols * patch_size, 3), dtype=np.uint8)
    for i, color in enumerate(patch_colors):
        r, c = divmod(i, cols)
        start_y = r * patch_size
        start_x = c * patch_size
        grid_img[start_y:start_y+patch_size, start_x:start_x+patch_size] = color
    return grid_img

grid_image = create_color_grid(detected_colors)

# === Step 5: Save the Grid Image ===
output_path = "Images/detected_patch_grid.jpg"
Image.fromarray(grid_image).save(output_path)
print(f"Saved detected patch grid to: {output_path}")

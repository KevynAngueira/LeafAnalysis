# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import cv2
import numpy as np
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
from colour.characterisation.datasets import CCS_COLOURCHECKERS
from colour.models import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES

# === CONFIGURATION ===
input_path = "Images/Reference.jpg"
output_base = "Images/output_pass"
num_passes = 10

# === ColorChecker Reference sRGB 255 values ===
reference_srgb_255 = np.array([
    [115, 82, 68],   [194, 150, 130], [98, 122, 157],  [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44],  [80, 91, 166],
    [193, 90, 99],   [94, 60, 108],   [157, 188, 64],  [224, 163, 46],
    [56, 61, 150],   [70, 148, 73],   [175, 54, 60],   [231, 199, 31],
    [187, 86, 149],  [8, 133, 161],   [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [85, 85, 85],    [52, 52, 52]
]) / 255.0

# === Functions ===
def srgb_to_linear(image):
    threshold = 0.04045
    return np.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(image):
    threshold = 0.0031308
    return np.where(image <= threshold, image * 12.92, 1.055 * (image ** (1 / 2.4)) - 0.055)

def apply_matrix_correction(image, matrix):
    flat = image.reshape((-1, 3))
    corrected = np.dot(flat, matrix.T)
    corrected = np.clip(corrected, 0, 1)
    return corrected.reshape(image.shape)

# === Load initial image ===
image = cv2.imread(input_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to 0-1
current_image = image_rgb.copy()

# === Start multi-pass correction ===
for i in range(num_passes):
    print(f"
--- PASS {i+1} ---")

    # Detect ColorChecker patches
    checkers = detect_colour_checkers_segmentation((current_image * 255).astype(np.uint8))
    if not checkers:
        raise ValueError("No ColorChecker detected in the image!")
    checker = checkers[0]

    # Get detected patch means
    patches = np.array(checker.data).reshape(24, -1, 3)
    detected_values = np.array([np.mean(patch, axis=0) for patch in patches])

    # Linearize both detected and reference values
    detected_lin = srgb_to_linear(detected_values)
    reference_lin = srgb_to_linear(reference_srgb_255)

    # Compute correction matrix
    M = colour.characterisation.matrix_colour_correction(detected_lin, reference_lin, method='Cheung 2004')
    print("Correction Matrix M:")
    print(M)

    # Linearize entire image
    image_lin = srgb_to_linear(current_image)

    # Apply correction
    corrected_lin = apply_matrix_correction(image_lin, M)

    # Convert back to sRGB
    corrected_srgb = linear_to_srgb(corrected_lin)
    corrected_srgb_clipped = np.clip(corrected_srgb, 0, 1)

    # Save for inspection
    out_path = f"{output_base}_{i+1}.png"
    cv2.imwrite(out_path, cv2.cvtColor((corrected_srgb_clipped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Prepare for next round
    current_image = corrected_srgb_clipped.copy()

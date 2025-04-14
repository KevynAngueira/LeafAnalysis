# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-04-14


import cv2
import numpy as np
from dataclasses import dataclass

from Scripts.HSVMask import HSVMask
from Scripts.LABMask import LABMask
from Scripts.ResizeForDisplay import resize_for_display

"""
@dataclass
class LeafSeparatorConfig:
    leaf_bounds: tuple = (np.array([100, 0, 0]), np.array([179, 255, 255]))
    low_sat_leaf_bounds: tuple = (np.array([0, 0, 0]), np.array([85, 255, 255]))
    sat_threshold: int = None
    target_dimensions: tuple = (650, 100)
    border_margin: int = 30
    kernel_size: tuple = (3, 3)
    morph_iterations: int = 2
    blur: tuple = (3, 3)
"""

@dataclass
class LeafSeparatorConfig:
    leaf_bounds: tuple = (np.array([0, 0, 110]), np.array([255, 255, 255]))
    target_dimensions: tuple = (650, 100)
    border_margin: int = 30
    kernel_size: tuple = (3, 3)
    morph_iterations: int = 2
    blur: tuple = (3, 3)


class LeafSeparator:
    def __init__(self, config: LeafSeparatorConfig=None):
        if config is None:
            config = LeafSeparatorConfig()

        self.leaf_bounds = config.leaf_bounds
        self.target_dimensions = config.target_dimensions
        self.border_margin = config.border_margin
        self.kernel_size = config.kernel_size
        self.morph_iterations = config.morph_iterations
        self.blur = config.blur
        
        self.leafMask = LABMask(config.leaf_bounds)
    
    def _crop_using_contours(self, image):
        """
        Crop the tool's frontpiece by detecting contours at the edges of the image and cropping to their mean height.
        """

        # Load the image (grayscale)
        height, width, _ = image.shape

        # Remove Noise
        if self.morph_iterations > 0:
            kernel = np.ones(self.kernel_size, np.uint8)
            morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        else:
            morphed = image

        blurred = cv2.GaussianBlur(morphed, self.blur, 0)

        gray = blurred[:, :, 0]

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Collect edge points
        left_points, right_points = [], []
        top_points, bottom_points = [], []

        for contour in contours:
            for point in contour[:, 0]:  # Extract (x, y) points
                x, y = point
                if x <= self.border_margin:
                    left_points.append(x)
                if x >= width - self.border_margin:
                    right_points.append(x)
                if y <= self.border_margin:
                    top_points.append(y)
                if y >= height - self.border_margin:
                    bottom_points.append(y)

        # Compute average values (fallback to original borders if no points found)
        left_bound = int(np.mean(left_points)) if left_points else 0
        right_bound = int(np.mean(right_points)) if right_points else width
        top_bound = int(np.mean(top_points)) if top_points else 0
        bottom_bound = int(np.mean(bottom_points)) if bottom_points else height

        # Crop the image
        cropped_image = image[top_bound:bottom_bound, left_bound:right_bound]

        return cropped_image

    def _imagePreprocessing(self, image):
        """
        Applying preprocessing to the image
            Crop Frontpiece -> Crops out remaining frontpiece from view window
            Resize -> Resize image View Window to standardized size
        """

        cropped_image = self._crop_using_contours(image)

        resized_image = cv2.resize(cropped_image, self.target_dimensions)

        return resized_image

    def Extract(self, image, display=False):
        """
        Extract the leaf-only mask, count leaf pixels, and calculate leaf percentage
        """

        preprocessed = self._imagePreprocessing(image)

        leaf_result, leaf_mask = self.leafMask.applyMask(preprocessed)
        leaf_pixels = np.count_nonzero(leaf_mask == 255)

        total_pixels = leaf_mask.size
        leaf_percentage = (leaf_pixels / total_pixels) * 100
        
        if display:
            print(f"Leaf Pixels: {leaf_pixels}")
            print(f"Total Pixels: {total_pixels}")
            print(f"Leaf Percentage: {leaf_percentage}")

            cv2.imshow("Original", resize_for_display(image))
            cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
            cv2.imshow("Leaf Result", resize_for_display(leaf_result))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return leaf_result, leaf_mask, leaf_pixels, leaf_percentage

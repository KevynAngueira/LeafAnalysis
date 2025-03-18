# Author: Your Name
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:43:18


import cv2
import numpy as np
from dataclasses import dataclass

from Scripts.HSVMask import HSVMask
from Scripts.ResizeForDisplay import resize_for_display
from Scripts.CropAndRotate import cropAndRotate

@dataclass
class ViewWindowConfig:
    tool_bounds: tuple = (np.array([0, 0, 0]), np.array([110, 255, 255]))
    low_sat_tool_bounds: tuple = (np.array([4, 0, 0]), np.array([172, 255, 255]))
    sat_threshold: int = 40
    target_aspect_ratio: float = 6.5
    aspect_ratio_tolerance: float = 1.0
    kernel_size: tuple = (5, 5)
    morph_iterations: int = 2
    blur: tuple = (5, 5)

class ViewWindow:
    def __init__(self, config: ViewWindowConfig = None):
        if config is None:
            config = ViewWindowConfig()

        self.tool_bounds = config.tool_bounds
        self.low_sat_tool_bounds = config.low_sat_tool_bounds
        self.sat_threshold = config.sat_threshold
        self.target_aspect_ratio = config.target_aspect_ratio
        self.aspect_ratio_tolerance = config.aspect_ratio_tolerance
        self.kernel_size = config.kernel_size
        self.morph_iterations = config.morph_iterations
        self.blur = config.blur
        
        self.toolMask = HSVMask(config.tool_bounds, config.sat_threshold, config.low_sat_tool_bounds)

    def __imagePreprocessing(self, image):
        """
        Applying image preprocessing
            Tool Mask -> Seperates out tool 
            Morphological Close -> Closes small gaps
            Gaussian Blur -> Smoothens out edges
        """

        # Tool Mask
        _, preprocessed = self.toolMask.applyHSVMask(image)

        # Morphological Close
        if self.morph_iterations > 0:
            kernel = np.ones(self.kernel_size, np.uint8)
            preprocessed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        # Gaussian Blur
        preprocessed = cv2.GaussianBlur(preprocessed, self.blur, 0)

        return preprocessed

    def __getContours(self, gray):
        """
        Get the contours on the image
        """

        edges = cv2.Canny(gray, 50, 150)

        # Find contours from Canny edges
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def __drawContours(self, image, contours, color=(255,255,0)):
        """
        Draw contours on the given image
        """    
        def get_color(i):
            normalized = int(255 * (i % 10)/10 )
            return (255 - normalized, 0, normalized)  # Blue for small, Red for large

        drawn_contours = image.copy()
        
        if color is None:
            for i, c in enumerate(contours):
                color = get_color(i)
                cv2.drawContours(drawn_contours, [c], -1, color, 2)        
        else:
            cv2.drawContours(drawn_contours, contours, -1, color, 2)

        return drawn_contours

    def __contoursToViewWindow(self, contours, mask, display=False):
        """
        Selects which contour represents the View Window based on the target dimension.

        The View Window is:
        (1) The largest contour that matches the target aspect ratio (within tolerance).
        (2) A contour surrounded by black in the mask.
        """

        target_rect = None
        target_box = None
        max_area = 0

        # Convert mask to BGR for visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        for contour in contours:
            # Get min area bounding box
            min_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(min_rect)
            box = np.intp(box)  # Convert to integer points
            w, h = min_rect[1]  

            aspect_ratio = max(w, h) / min(w, h) if min(w,h) != 0 else 0

            # Capture the largest contour that matches the target aspect ratio (within tolerance)
            if abs(aspect_ratio - self.target_aspect_ratio) <= self.aspect_ratio_tolerance:
                area = w*h
                if area > max_area:
                    # Compute expanded bounding box
                    expanded_box = self.__expandRotatedBox(box, padding=10)

                    # Check if contour is surrounded by white area
                    if self.__isSurroundedByBlack(mask, expanded_box, box):
                        max_area = area
                        target_box = box
                        target_rect = min_rect

                    if display:
                        # Draw contour (green)
                        cv2.drawContours(mask_vis, [box], -1, (0, 255, 0), 2)
                        # Draw expanded bounding box (red)
                        cv2.drawContours(mask_vis, [expanded_box], -1, (0, 0, 255), 2)

                        cv2.imshow("Mask Vis", resize_for_display(mask_vis))
    
        return target_box, target_rect

    def __isSurroundedByBlack(self, mask, expanded_box, original_box, white_threshold=0.15):
        """
        Checks if a rotated contour (expanded_box) is surrounded by white areas in the mask.
        """

        # Create a blank mask of the same size as the input mask
        expanded_mask = np.zeros_like(mask, dtype=np.uint8)

        # Fill the expanded area with white
        cv2.fillPoly(expanded_mask, [expanded_box], 255)

        # Remove the original contour from the expanded area
        cv2.fillPoly(expanded_mask, [original_box], 0)

        # Get only the expanded area from the original mask
        surrounding_area = cv2.bitwise_and(mask, expanded_mask)

        # Compute percentage of white pixels in the surrounding area
        white_pixels = np.sum(surrounding_area == 255)
        total_pixels = np.sum(expanded_mask == 255)
        
        if total_pixels == 0:
            return False  # Prevent division by zero

        white_ratio = white_pixels / total_pixels
       
        return white_ratio < white_threshold  # At least 85% black surrounding the contour

    def __expandRotatedBox(self, box, padding=10):
        """
        Expands a rotated bounding box outward by a given padding amount.
        """

        # Compute the center of the box
        center = np.mean(box, axis=0)

        # Compute vectors for the box edges
        vectors = box - center

        # Normalize and scale vectors outward
        expanded_vectors = vectors * (1 + padding / np.linalg.norm(vectors, axis=1, keepdims=True))

        # Compute new expanded box coordinates
        expanded_box = np.intp(center + expanded_vectors)

        return expanded_box

    def Extract(self, image, display=False):
        """
        Extract the view window from the image
        """

        preprocessed = self.__imagePreprocessing(image)

        contours = self.__getContours(preprocessed)

        target_box, target_rect = self.__contoursToViewWindow(contours, preprocessed, display)

        view_window = cropAndRotate(image, target_rect)

        if display:
            all_contours = self.__drawContours(image, contours, None)
            target_countour = self.__drawContours(image, [target_box], (0, 255, 255))

            cv2.imshow("Original", resize_for_display(image))
            cv2.imshow("Preprocessed", resize_for_display(preprocessed))
            cv2.imshow("All Contours", resize_for_display(all_contours))
            cv2.imshow("Target Contour", resize_for_display(target_countour))
            cv2.imshow("View Window", resize_for_display(view_window))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return view_window


# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20


import cv2
import numpy as np
from dataclasses import dataclass
from itertools import combinations

from Scripts.HSVMask import HSVMask
from Scripts.LABMask import LABMask

from Scripts.ResizeForDisplay import resize_for_display
from Scripts.CropAndRotate import cropAndRotate

@dataclass
class ViewWindowConfig:
    tool_bounds: tuple = (np.array([165, 130, 85]), np.array([255, 170, 255]))
    target_aspect_ratio: float = 6.5
    aspect_ratio_tolerance: float = 0.8
    kernel_size: tuple = (5, 5)
    morph_iterations: int = 2
    blur: tuple = (5, 5)

class ViewWindow:
    def __init__(self, config: ViewWindowConfig = None):
        if config is None:
            config = ViewWindowConfig()

        self.tool_bounds = config.tool_bounds
        self.target_aspect_ratio = config.target_aspect_ratio
        self.aspect_ratio_tolerance = config.aspect_ratio_tolerance
        self.kernel_size = config.kernel_size
        self.morph_iterations = config.morph_iterations
        self.blur = config.blur
        
        self.toolMask = LABMask(config.tool_bounds)

    def _imagePreprocessing(self, image):
        """
        Applying image preprocessing
            Tool Mask -> Seperates out tool 
            Morphological Close -> Closes small gaps
            Gaussian Blur -> Smoothens out edges
        """

        # Tool Mask
        _, preprocessed = self.toolMask.applyMask(image)

        # Morphological Close
        if self.morph_iterations > 0:
            kernel = np.ones(self.kernel_size, np.uint8)
            preprocessed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        # Gaussian Blur
        preprocessed = cv2.GaussianBlur(preprocessed, self.blur, 0)

        return preprocessed

    def _getContours(self, gray):
        """
        Get the contours on the image
        """

        edges = cv2.Canny(gray, 50, 150)

        # Find contours from Canny edges
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def _drawContours(self, image, contours, color=(255,255,0)):

        image_shape = image.shape

        def get_color(i):
            normalized = int(255 * (i % 10)/10 )
            return (255 - normalized, 0, normalized)

        drawn_contours = image.copy()

        if color is None:
            for i, contour in enumerate(contours):
                if contour is None or len(contour) < 4:
                    continue
                            
                draw_color = get_color(i) if color is None else color
                cv2.drawContours(drawn_contours, [contour], -1, draw_color, 2)
        else:
            cv2.drawContours(drawn_contours, contours, -1, color, 2)


        return drawn_contours
        
    def _contoursToViewWindow(self, contours, mask, display=False, fallback = True, fallback_top_k=3):
        """
        Attempts to detect the view window from a list of contours
        (1) First tries finding a "direct match" (largest rect matching aspect ratio and surrounded by white)
        (2) Then tries fallback to scaled rect (eliminated protrusions, then largest rect surrounded by white)
        """

        target_box = None
        target_rect = None
        max_area = 0

        fallback_candidates = []

        # Visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            if len(contour) < 4:
                continue  # Skip trivial or broken contours

            min_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(min_rect).astype(np.int32)
            w, h = min_rect[1]

            if w == 0 or h == 0: 
                continue # Skip trivial 2D contours

            area = w * h
            aspect_ratio = max(w, h) / min(w, h)

            # Track for fallback later
            fallback_candidates.append((area, contour))

            # Try direct match
            if abs(aspect_ratio - self.target_aspect_ratio) <= self.aspect_ratio_tolerance:
                expanded_box = self.__expandRotatedBox(min_rect, padding=20)

                if self.__isSurroundedByWhite(mask, expanded_box, box) and area > max_area:
                    target_box = box
                    target_rect = min_rect
                    max_area = area

                if display:
                    cv2.drawContours(mask_vis, [box], -1, (0, 255, 0), 2)
                    cv2.drawContours(mask_vis, [expanded_box], -1, (0, 0, 255), 2)

        # Fallback: if no valid contour was found
        if target_box is None and fallback and fallback_candidates:
            # Try top-K largest contours
            fallback_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by area descending

            for _, contour in fallback_candidates[:fallback_top_k]:
                best_box, best_rect = self._find_best_scaled_rect(contour, mask.shape)
                
                if best_box is not None:
                    expanded_box = self.__expandRotatedBox(best_rect, padding=20)
                    
                    if self.__isSurroundedByWhite(mask, expanded_box, best_box):
                        area = cv2.contourArea(best_box)
                        
                        if area > max_area:
                            target_box = best_box
                            target_rect = best_rect
                            max_area = area

        if display and target_box is not None:
            cv2.drawContours(mask_vis, [target_box], -1, (255, 0, 0), 2)
            cv2.imshow("Mask Vis", resize_for_display(mask_vis))
            cv2.waitKey(1)

        return target_box, target_rect

    def _percentile_contour_width_mask(self, contour, image_shape, percentile=75):
        """
        Calculates the specified percentile width of a contour using a binary mask.
        """
        # Step 1: Draw the contour to a binary mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Step 2: Scan each row (y-axis) for nonzero (white) pixels
        heights, widths = mask.shape
        row_widths = []
        for y in range(heights):
            x_coords = np.where(mask[y, :] > 0)[0]
            if x_coords.size >= 2:
                row_widths.append(x_coords.max() - x_coords.min())

        return np.percentile(row_widths, percentile) if row_widths else 0

    def _find_best_scaled_rect(self, contour, image_shape, target_aspect_ratio=6.5):
        """
        Given a contour, find the best-placed scaled rectangle that fits inside it
        by maximizing overlap.
        
        Args:
            contour: The original contour (N, 1, 2)
            image_shape: Shape of the full image for mask creation
            target_aspect_ratio: Width / Height ratio (default 6.5)
        
        Returns:
            best_box: Rotated rectangle box (4 points) with best overlap
        """

        # Step 1: Get minAreaRect
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # Step 2: Ensure w is width and h is height such that width > height
        if w < h:
            w, h = h, w
            angle += 90.0

        # Step 3: Estimate scaled rectangle
        # Use 75th percentile of all segment widths
        scale_width = self._percentile_contour_width_mask(contour, image_shape, 75)
        scale_height = scale_width / target_aspect_ratio

        # Create the rotated rectangle of target size
        scaled_rect = ((cx, cy), (scale_width, scale_height), angle)

        # Step 4: Slide the rectangle vertically and find best overlap
        contour_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        max_overlap = 0
        best_box = None
        best_rect = None
        step = 2  # pixels to move per step
        search_range = int(h // 2)

        for dy in range(-search_range, search_range + 1, step):
            moved_center = (cx, cy + dy)
            moved_rect = (moved_center, (scale_width, scale_height), angle)
            box = cv2.boxPoints(moved_rect).astype(np.int32)

            rect_mask = np.zeros_like(contour_mask)
            cv2.drawContours(rect_mask, [box], -1, 255, thickness=cv2.FILLED)

            overlap = cv2.countNonZero(cv2.bitwise_and(rect_mask, contour_mask))
            if overlap > max_overlap:
                max_overlap = overlap
                best_box = box
                best_rect = moved_rect

        return best_box, best_rect

    def __isSurroundedByWhite(self, mask, expanded_box, original_box, white_threshold=0.75):
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

        #print(f"White Pixels: {white_pixels}")
        #print(f"Total Pixels: {total_pixels}")

        if total_pixels == 0:
            return False  # Prevent division by zero

        white_ratio = white_pixels / total_pixels

        #print(f"White Ratio: {white_ratio}")

        return white_ratio > white_threshold  # At least 85% black surrounding the contour

    def __expandRotatedBox(self, rect, padding=20):
        """
        Expands a rotated bounding box outward by a given padding amount.
        Assumes 'box' is a 4x2 array of points (clockwise or counter-clockwise).
        """

        # Unpack the rect
        (center, (width, height), angle) = rect

        # Expand width and height
        expanded_width = width + 2 * padding
        expanded_height = height + 2 * padding

        # Construct new expanded rect
        expanded_rect = (center, (expanded_width, expanded_height), angle)

        # Get corner points of the expanded box
        expanded_box = cv2.boxPoints(expanded_rect)
        expanded_box = np.intp(expanded_box) 

        return expanded_box

    def Extract(self, image, display=False, **kwargs):
        """
        Extract the view window from the image
        """
        preprocessed = self._imagePreprocessing(image)

        contours = self._getContours(preprocessed)

        target_box, target_rect = self._contoursToViewWindow(contours, preprocessed, display)

        view_window = cropAndRotate(image, target_rect)

        if display:
            all_contours = self._drawContours(image, contours, None)
            target_countour = self._drawContours(image, [target_box], (0, 255, 255))

            cv2.imshow("Original", resize_for_display(image))
            cv2.imshow("Preprocessed", resize_for_display(preprocessed))
            cv2.imshow("All Contours", all_contours)
            cv2.imshow("Target Contour", resize_for_display(target_countour))
            cv2.imshow("View Window", resize_for_display(view_window))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return view_window


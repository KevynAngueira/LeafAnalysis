# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-09-22

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from LeafScan.Utils import resize_for_display

class FOpticalFlowDetector:
    def __init__(self, slice_height=100, band_height=40, empty_frame_threshold=0.02):
        self.slice_height = slice_height
        self.band_height = band_height
        self.empty_frame_threshold = empty_frame_threshold

        center_y = slice_height//2
        self.template_start_y = max(0, center_y - (band_height // 2))
        self.template_end_y = min(slice_height, center_y + (band_height // 2))

    def reset(self):
        self.total_displacement = 0
        self.frame_count = 0

        self.prev_image = None
        self.prev_mask = None
        self.prev_max_loc = None

        self.cummulative_displacements = []

    def _imagePreprocessing(self, image):
        """
        Applying image preprocessing 
            Grayscale -> Convert image to grayscale
            Mask -> Eliminate non-target areas
            CLAHE -> Improve local contrast to boost subtle differences
            Sharpen -> Further instensify differences
        """

        resized_image = cv2.resize(image, (650, 100))

        # Grayscale -> Convert image to grayscale
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # CLAHE -> Improve local contrast to boost subtle differences
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpen -> Further instensify differences
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        sharpened_float = sharpened.astype(np.float32) / 255.0
        
        return sharpened_float

    def __extractTemplate(self, image):
        """
        Extracts a template from the image. 
        The template is a horizontal band from the center of the image.
        """

        # Extract the band from image and mask
        band = image[self.template_start_y:self.template_end_y, :]

        return band

    def _checkEmptyFrame(self, mask):
        nonzero_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        nonzero_ratio = nonzero_pixels / total_pixels

        return nonzero_ratio < self.empty_frame_threshold        

    def visualize_flow(self, flow):
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return resize_for_display(bgr)

    def trackFrameDisplacement(self, image, mask):
        preprocessed = self._imagePreprocessing(image)

        if self.frame_count % 1 == 0:
            if self.prev_image is None:
                self.prev_image = preprocessed
                return 0, image

            if self._checkEmptyFrame(mask):
                self.prev_image = preprocessed
                return 0, image

            # Calculate optical flow
            prev_gray = (self.prev_image * 255).astype(np.uint8)
            curr_gray = (preprocessed * 255).astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            vertical_flow = flow[..., 1]
            band_flow = vertical_flow[self.template_start_y:self.template_end_y, :]

            # Threshold: suppress small flow values
            abs_band_flow = np.abs(band_flow)
            threshold_mask = abs_band_flow >= 0.8  # Ignore subpixel noise
            filtered_flow = band_flow[threshold_mask]

            if filtered_flow.size == 0:
                median_displacement = 0
            else:
                median_displacement = np.mean(filtered_flow)

            self.prev_image = preprocessed

            # Optional visualization
            flow_vis = self.visualize_flow(flow)

            return median_displacement, flow_vis

        return 0, preprocessed


    def trackCummulativeDisplacement(self, image, mask):
        """
        Track the cumulative displacement across the video
        """
        frame_displacement, drawn_template = self.trackFrameDisplacement(image, mask)

        self.cummulative_displacements.append(frame_displacement)
        self.frame_count += 1

        return frame_displacement, drawn_template

    def getSliceIndexes(self, remaining_leaf_length):
        """
        Returns the frame indexes of the unique leaf slices.
        This is done by scaling the cumulative vertical movement to the remaining leaf length
        """
        
        if not self.cummulative_displacements:
            return []

        # Determine majority direction (+ or -)
        total = sum(self.cummulative_displacements)
        majority_sign = 1 if total >= 0 else -1

        # Filter displacements by majority direction and sub-pixel movement
        #filtered_displacements = [
        #    d*majority_sign for d in self.cummulative_displacements
        #]
        filtered_displacements = [
            d*majority_sign if d * majority_sign > 0 else 0
            for d in self.cummulative_displacements
        ]

        # Compute cumulative sum of filtered values
        cumulative_sum = np.cumsum(filtered_displacements)

        # Calculate the height of each slice by scaling displacement to remaining leaf length
        if remaining_leaf_length is None:
            slice_height = self.slice_height
        else:
            number_of_slices = math.ceil(remaining_leaf_length) - 1
            total_positive_disp = cumulative_sum[-1] if cumulative_sum.any() else 1
            slice_height = total_positive_disp / number_of_slices
        
        # Extract slices based on calculated slice height
        displacement_threshold = 0
        slice_indexes = []

        for idx, disp_sum in enumerate(cumulative_sum):
            if disp_sum > displacement_threshold: 
                print(f"Detected slice {len(slice_indexes)}: idx = {idx} | displacement = {disp_sum}")
                slice_indexes.append(idx)
                displacement_threshold += slice_height

        print(slice_indexes)
        return slice_indexes
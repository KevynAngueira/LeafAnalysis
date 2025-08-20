# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import os
import cv2
import numpy as np

import sys
from dataclasses import dataclass, field

sys.path.append("..")

from Scripts.ResizeForDisplay import resize_for_display
from Scripts.CropAndRotate import cropAndRotate
from Scripts.LABMask import LABMask

IMG_PATH = "./flatboard.jpg"

@dataclass
class LeafDetectionConfig:
    leaf_targets: tuple =  (
        30,    # green-ish
        #np.array([128, 128])   # brown-ish
    )
    target_tolerance: int = 100
    percent_tolerance: float = 0.02

class LeafDetector: 

    def __init__(self, leaf_config: LeafDetectionConfig = None):
        if leaf_config is None:
            leaf_config = LeafDetectionConfig()
        
        self.__dict__.update(vars(leaf_config))

    def meanLabDistance(self, image_lab, contour, target, max_empty_ratio=0.5):
        # Step 1: Create mask for this contour
        mask_contour = np.zeros(image_lab.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_contour, [contour], -1, 255, -1)

        # Step 2: Create mask of valid (non-black) pixels (e.g., L > 0)
        non_black = cv2.inRange(image_lab, (1, 0, 0), (255, 255, 255))

        # Step 3: Calculate number of pixels inside contour and number of valid pixels
        total_pixels = cv2.countNonZero(mask_contour)
        valid_pixels = cv2.countNonZero(cv2.bitwise_and(mask_contour, non_black))

        # Step 4: Check empty ratio
        if total_pixels == 0 or (1 - valid_pixels / total_pixels) > max_empty_ratio:
            return float('inf')

        # Step 5: Compute mean over valid pixels only
        combined_mask = cv2.bitwise_and(mask_contour, non_black)
        mean = cv2.mean(image_lab, mask=combined_mask)[:3]
        mean_a = mean[1]

        return abs(mean_a - target)

    def _isolateTarget(self, image, target_contour):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [target_contour], -1, 255, -1)
        isolated = cv2.bitwise_and(image, image, mask=mask)
        return isolated

    def stack_leaf_segments_vertically(self, leaf_segments, padding=10, bg_color=0):
        # Step 1: Sort segments by x-position of contour center
        def x_center(contour):
            M = cv2.moments(contour)
            return M['m10'] / M['m00'] if M['m00'] != 0 else 0

        sorted_segments = sorted(leaf_segments, key=lambda seg: x_center(seg[2]))

        # Step 2: Determine dimensions for stacked image
        widths = [img.shape[1] for img, _, _ in sorted_segments]
        heights = [img.shape[0] for img, _, _ in sorted_segments]
        max_width = max(widths)
        total_height = sum(heights) + padding * (len(sorted_segments) - 1)

        # Step 3: Create empty black canvas
        stacked_image = np.full((total_height, max_width, 3), bg_color, dtype=np.uint8)

        # Step 4: Paste each segment centered horizontally, stacked vertically
        y_offset = 0
        for img, _, _ in sorted_segments:
            h, w = img.shape[:2]
            x_offset = (max_width - w) // 2
            stacked_image[y_offset:y_offset + h, x_offset:x_offset + w] = img
            y_offset += h + padding

        return stacked_image

    def detect_leaf_segments(self, image_lab, contours):
        #image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        leaf_like = []
        w, h = image_lab.shape[:2]
        total_area = w*h
        #print(total_area * self.percent_tolerance)

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            #min_rect = cv2.minAreaRect(cnt)
            #w,h = min_rect[1]
            #rect_area = w*h

            solidity = area / total_area
            if solidity < self.percent_tolerance:
                #print("Skipping, size")
                continue

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            #print(f"Contour_{idx}: {area} | {len(approx)}") #| {rect_area}")
            if len(approx) < 4:  
                #print("Skipping, complexity")
                continue

            min_dist = min(self.meanLabDistance(image_lab, cnt, target) for target in self.leaf_targets)
            print(min_dist, area)

            if min_dist < self.target_tolerance:  # adjustable threshold
                leaf_like.append((cnt, area))

        leaf_segments = []

        counter = 0
        for cnt, _ in leaf_like:
            x, y, w, h = cv2.boundingRect(cnt)
            isolated = self._isolateTarget(image_lab, cnt)
            leaf_crop = isolated[y:y+h, x:x+w]
            
            print(f"==== Contour_{counter} ====")

            contour_area = cv2.contourArea(cnt) 
            print("Contour Area:", contour_area)

            # Step 1: Create mask for this contour
            mask_contour = np.zeros(isolated.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_contour, [cnt], -1, 255, -1)

            # Step 2: Create mask of valid (non-black) pixels (e.g., L > 0)
            non_black = cv2.inRange(isolated, (1, 0, 0), (255, 255, 255))

            # Step 3: Calculate number of pixels inside contour and number of valid pixels
            leaf_mask = cv2.bitwise_and(mask_contour, non_black)
            valid_pixels = cv2.countNonZero(leaf_mask)
            leaf_mask = leaf_mask[y:y+h, x:x+w]

            print("Contour Pixels:", valid_pixels)

            leaf_segments.append((leaf_crop, leaf_mask, cnt))

            counter += 1
        
        if len(leaf_segments) > 0: 
            stacked_image = self.stack_leaf_segments_vertically(leaf_segments)
        else:
            stacked_image = image_lab

        return stacked_image, leaf_segments  

 

# Author: Kevyn Angueira Irizarry
# Created: 2025-06-25
# Last Modified: 2025-06-25

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
    target_tolerance: int = 80
    percent_tolerance: float = 0.02

class LeafDetector: 

    def __init__(self, leaf_config: LeafDetectionConfig = None):
        if leaf_config is None:
            leaf_config = LeafDetectionConfig()
        
        self.__dict__.update(vars(leaf_config))

    def mean_lab_distance(self, image_lab, contour, target, max_empty_ratio=0.5):
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

            if area < total_area * self.percent_tolerance:
                #print("Skipping, size")
                continue

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            #print(f"Contour_{idx}: {area} | {len(approx)}") #| {rect_area}")
            if len(approx) < 4:  
                #print("Skipping, complexity")
                continue

            min_dist = min(self.mean_lab_distance(image_lab, cnt, target) for target in self.leaf_targets)
            #print(min_dist)

            if min_dist < self.target_tolerance:  # adjustable threshold
                leaf_like.append((cnt, area))

        # Sort left to right (assuming horizontal layout)
        def x_center(cnt): 
            M = cv2.moments(cnt[0])
            return M["m10"] / M["m00"] if M["m00"] != 0 else 0

        leaf_like.sort(key=lambda c: x_center(c))

        leaf_images = []

        counter = 0
        for cnt, _ in leaf_like:
            x, y, w, h = cv2.boundingRect(cnt)
            isolated = self._isolateTarget(image_lab, cnt)
            #leaf_crop = isolated[y:y+h, x:x+w]
            leaf_images.append((isolated, cnt))
            
            print(f"==== Contour_{counter} ====")

            contour_area = cv2.contourArea(cnt) 
            print("Contour Area:", contour_area)

            # Step 1: Create mask for this contour
            mask_contour = np.zeros(image_lab.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_contour, [cnt], -1, 255, -1)

            # Step 2: Create mask of valid (non-black) pixels (e.g., L > 0)
            non_black = cv2.inRange(image_lab, (1, 0, 0), (255, 255, 255))

            # Step 3: Calculate number of pixels inside contour and number of valid pixels
            valid_pixels = cv2.countNonZero(cv2.bitwise_and(mask_contour, non_black))
            print("Contour Pixels:", valid_pixels)

            counter += 1





        return leaf_images    

 

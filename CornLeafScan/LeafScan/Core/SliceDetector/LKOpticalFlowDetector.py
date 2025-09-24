# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from LeafScan.Utils import resize_for_display

class LKOpticalFlowDetector:
    def __init__(self, slice_height=100, band_height=40, empty_frame_threshold=0.02):
        self.slice_height = slice_height
        self.band_height = band_height
        self.empty_frame_threshold = empty_frame_threshold

        center_y = slice_height//2
        self.template_start_y = max(0, center_y - (band_height // 2))
        self.template_end_y = min(slice_height, center_y + (band_height // 2))

        self.lk_params = dict(winSize=(15, 15),
                          maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def reset(self):
        self.total_displacement = 0
        self.frame_count = 0

        self.prev_image = None
        self.prev_mask = None
        self.prev_max_loc = None

        self.cummulative_displacements = []

        self.prev_points = None

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
        
        '''
        cv2.imshow("Image", resize_for_display(image))
        cv2.imshow("Preprocessed", resize_for_display(sharpened_float))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
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
   
    def __getFeaturePoints(self, image):
        """
        Extract good features to track from the band in the image.
        """
        band = image[self.template_start_y:self.template_end_y, :]
        points = cv2.goodFeaturesToTrack(band, mask=None, maxCorners=50, qualityLevel=0.01, minDistance=5)
        
        if points is not None:
            # Adjust y-coordinates to match full image
            points[:, 0, 1] += self.template_start_y

        return points

    def __lucasKanadeTracking(self, prev_img, curr_img, prev_pts):
        """
        Tracks feature points using Lucas-Kanade optical flow.
        """
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **self.lk_params)
        
        # Filter only valid points
        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]
        
        return valid_prev, valid_curr

    def trackFrameDisplacement(self, image, mask):
        preprocessed = self._imagePreprocessing(image)
        gray = (preprocessed * 255).astype(np.uint8)

        if self.frame_count % 1 == 0:

            if self.prev_image is None:
                self.prev_image = gray
                self.prev_points = self.__getFeaturePoints(gray)
                return 0, image

            if self._checkEmptyFrame(mask):
                self.prev_image = gray
                self.prev_points = self.__getFeaturePoints(gray)
                return 0, image

            if self.prev_points is None or len(self.prev_points) < 3:
                self.prev_image = gray
                self.prev_points = self.__getFeaturePoints(gray)
                return 0, image

            prev_pts, curr_pts = self.__lucasKanadeTracking(self.prev_image, gray, self.prev_points)

            if len(prev_pts) == 0:
                self.prev_image = gray
                self.prev_points = self.__getFeaturePoints(gray)
                return 0, image

            # Calculate vertical displacement only
            y_displacements = prev_pts[:, 1] - curr_pts[:, 1]
            vertical_disp = np.sum(y_displacements)  # sum seems more stable than mean for you

            # Visualization (optional)
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for (p, q) in zip(prev_pts, curr_pts):
                p = tuple(p.astype(int))
                q = tuple(q.astype(int))
                cv2.line(debug_img, p, q, (0, 255, 0), 2)
                cv2.circle(debug_img, q, 3, (0, 0, 255), -1)

            # Update state
            self.prev_image = gray
            self.prev_points = self.__getFeaturePoints(gray)

            #print(vertical_disp)

            return vertical_disp, debug_img

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
        '''
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
        '''
        
        # Compute cumulative sum of filtered values
        #cumulative_sum = np.cumsum(filtered_displacements)
        cumulative_sum = np.cumsum(self.cummulative_displacements)

        # Calculate the height of each slice by scaling displacement to remaining leaf length
        if remaining_leaf_length is None:
            slice_height = self.slice_height
        else:
            number_of_slices = math.ceil(remaining_leaf_length) - 1
            total_positive_disp = abs(cumulative_sum[-1] if cumulative_sum.any() else 1)
            slice_height = total_positive_disp / number_of_slices
        
        # Extract slices based on calculated slice height
        displacement_threshold = 0
        slice_indexes = []

        for idx, disp_sum in enumerate(cumulative_sum):
            if abs(disp_sum) > displacement_threshold: 
                print(f"Detected slice {len(slice_indexes)}: idx = {idx} | displacement = {disp_sum}")
                slice_indexes.append(idx)
                displacement_threshold += slice_height

        print(slice_indexes)
        return slice_indexes

    def _smoothDisplacements(self, displacements, window_size=5):
        if len(displacements) < window_size:
            return np.array(displacements)

        padded = np.pad(displacements, (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        return smoothed
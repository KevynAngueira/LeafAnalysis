# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 16:13:31


import cv2
import numpy as np

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig
from Scripts.CropAndRotate import cropAndRotate

class StabilizedViewWindow(ViewWindow):
    def __init__(self, config: ViewWindowConfig = None, movement_thresholds=(5,10,30), alpha: float = 0.1):
        super().__init__(config)
        
        self.prev_rect = None
        self.prev_center = None

        self.frame_lost_count = 0
        self.frame_skip_count = 0 

        self.movement_thresholds = movement_thresholds
        self.alpha = alpha

    def __calculateCenter(self, rect):
        """Calculate the center of the bounding box (minAreaRect)."""
        center = rect[0]  # The center of the bounding box (minAreaRect)
        return np.array(center)

    def __stabilizeMovement(self, current_rect):
        """
        Stabilize movement by comparing the current center to the previous center.
        Small changes are ignored, medium changes are accepted, and large changes are checked over multiple frames.
        """
        
        current_center = self.__calculateCenter(current_rect)

        if self.prev_center is None:
            self.prev_center = current_center
            self.prev_rect = current_rect
            return current_rect
        
        displacement = np.linalg.norm(current_center - self.prev_center)

        # Define thresholds for small, medium, and large movements
        small_threshold = self.movement_thresholds[0]
        medium_threshold = self.movement_thresholds[1]  
        large_threshold = self.movement_thresholds[2]

        if displacement < small_threshold:
            # No significant movement, stabilize by keeping the previous center
            return self.prev_rect
        elif displacement < medium_threshold:
            # Moderate movement, accept the new center
            self.prev_rect = current_rect
            self.prev_center = current_center
            print(f"medium Movemet: {self.prev_center}")
            return current_rect
        else:
            # Large movement, check over several frames
            self.frame_skip_count += 1
            # If the large movement is confirmed after a few frames, accept the new center
            if self.frame_skip_count >= 3:
                print("++++++++++++++++++++++++++++++++++++++++")
                print(f"LARGE Movemet: {self.prev_center}")
                print("++++++++++++++++++++++++++++++++++++++++")
                self.prev_rect = current_rect
                self.prev_center = current_center
                self.frame_skip_count = 0
                return current_rect
            else:
                print(f"Large Movement -- Skip {self.frame_skip_count}")
                # Skip the large movement for now
                return self.prev_rect
        
    def Extract(self, image, display=False):
        """
        Extract the stabilized view window from the image, based on center displacement.
        """

        preprocessed = self._imagePreprocessing(image)
        contours = self._getContours(preprocessed)

        # Find the target contour and its minAreaRect
        target_box, target_rect = self._contoursToViewWindow(contours, preprocessed, display)

        if target_box is not None:
            # Case: Target is found 

            #current_center = self.__calculateCenter(target_rect)

            # Stabilize the movement based on center displacement
            stabilized_rect = self.__stabilizeMovement(target_rect)

            # Use the stabilized center to crop and rotate the image
            view_window = cropAndRotate(image, stabilized_rect)

            self.frame_lost_count = 0

            if display:
                # Draw contours and stabilized view window
                all_contours = self._drawContours(image, contours, None)
                target_countour = self._drawContours(image, [target_box], (0, 255, 255))

                cv2.imshow("Original", resize_for_display(image))
                cv2.imshow("Preprocessed", resize_for_display(preprocessed))
                cv2.imshow("All Contours", resize_for_display(all_contours))
                cv2.imshow("Target Contour", resize_for_display(target_countour))
                cv2.imshow("View Window", resize_for_display(view_window))

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return view_window
        
        elif self.prev_rect is not None:
            # Case: Target is temporarily lost, assume no movement
            self.frame_lost_count += 1
            if self.frame_lost_count >= 3:
                self.prev_rect = None
                self.prev_center = None
            prev_window = cropAndRotate(image, self.prev_rect)
            return prev_window
        else:
            # Case: Target is unknown
            empty_image = np.zeros((100, 650, 3), dtype=np.uint8)
            return empty_image

            
            
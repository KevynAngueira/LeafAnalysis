# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-03-20


import cv2
import numpy as np

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig
from Scripts.ResizeForDisplay import resize_for_display
from Scripts.CropAndRotate import cropAndRotate

class StabilizedViewWindow(ViewWindow):
    def __init__(self, config: ViewWindowConfig = None, movement_threshold=20, alpha: float = 0.1, confirmation_frames=5):
        super().__init__(config)
        
        self.prev_rect = None
        self.prev_center = None
        self.prev_window = None

        self.movement_threshold = movement_threshold
        self.alpha = alpha

        self.move_confirmation_counter = 0 
        self.lost_confirmation_counter = 0

        self.confirmation_frames = confirmation_frames

    def __calculateCenter(self, rect):
        """Calculate the center of the bounding box (minAreaRect)."""
        center = rect[0]  # The center of the bounding box (minAreaRect)
        return np.array(center)

    def __ema(self, prev_value, new_value, alpha):
        """Exponential Moving Average smoothing function."""
        return (1 - alpha) * prev_value + alpha * new_value
    
    def __smoothDisplacement(self, current_rect, alpha=None):
        if alpha is None:
            alpha = self.alpha      

        prev_center, prev_size, prev_angle = self.prev_rect
        curr_center, curr_size, curr_angle = current_rect

        is_swapped = (curr_angle % 180) >= 90

        if is_swapped:
            curr_size = curr_size[::-1]
            curr_angle = (curr_angle % 180) - 90

        smoothed_center = tuple(self.__ema(np.array(prev_center), np.array(curr_center), alpha))
        smoothed_size = tuple(self.__ema(np.array(prev_size), np.array(curr_size), alpha))
        smoothed_angle = self.__ema(prev_angle, curr_angle, alpha)

        smoothed_rect = (smoothed_center, smoothed_size, smoothed_angle)

        if is_swapped:
            swapped_size = smoothed_size[::-1]
            swapped_angle = smoothed_angle + 90
        else:
            swapped_size = smoothed_size
            swapped_angle = smoothed_angle
        
        stabilized_rect = (smoothed_center, swapped_size, swapped_angle)

        self.prev_rect = smoothed_rect
        self.prev_center = self.__calculateCenter(stabilized_rect)

        return stabilized_rect

    def __stabilizeMovement(self, current_rect):
        """
        Stabilize movement by comparing the current center to the previous center.
        Small changes are ignored, medium changes are accepted, and large changes are checked over multiple frames.
        """

        current_center = self.__calculateCenter(current_rect)

        if self.prev_center is None:
            # First frame, initialize values
            self.prev_center = current_center
            self.prev_rect = current_rect
            return current_rect
        
        displacement = np.linalg.norm(current_center - self.prev_center)

        if displacement < self.movement_threshold:
            # Small movement → Apply EMA to position, size, and 
            #print("Small Movement")
            stabilized_rect = self.__smoothDisplacement(current_rect)
            self.move_confirmation_counter = max(self.move_confirmation_counter // 2, 0) 
            return stabilized_rect

        else:
            # Large movement → Require confirmation over multiple frames
            self.move_confirmation_counter += 1
            
            #print(f"Large Movement -> Frame {self.move_confirmation_counter}")
            
            if self.move_confirmation_counter >= self.confirmation_frames:
                # Confirm large movement, then smoothly transition over multiple frames
                alpha = min(self.alpha * (self.move_confirmation_counter / self.confirmation_frames), 0.95)
               
                # Confirmed large movement, apply EMA
                stabilized_rect = self.__smoothDisplacement(current_rect, alpha)

                #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                #print(f"Applying Large Move: {stabilized_rect}")
                #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                return stabilized_rect
            else:
                # Hold previous rect until confirmation
                return self.prev_rect
        
    def __handleLostTarget(self):
        """
        Handle the target view window is lost.
        """

        if self.prev_rect is not None:
            # Lost Target → Require confirmation over multiple frames
            self.lost_confirmation_counter += 1
            
            if self.lost_confirmation_counter >= self.confirmation_frames:
                # Confirmed target is lost, set view_window to empty
                self.prev_rect = None
                self.prev_center = None
                self.lost_confirmation_counter = 0
            
            return self.prev_window
        else:
            empty_image = np.zeros((100, 650, 3), dtype=np.uint8)
            return empty_image
                

    def Extract(self, image, display=False):
        """
        Extract the stabilized view window from the image, based on center displacement.
        """

        preprocessed = self._imagePreprocessing(image)
        contours = self._getContours(preprocessed)

        # Find the target contour and its minAreaRect
        target_box, target_rect = self._contoursToViewWindow(contours, preprocessed, display)

        if target_box is not None:
            # Target Found → Stabilize View Window
            stabilized_rect = self.__stabilizeMovement(target_rect)
            self.lost_confirmation_counter = 0

            view_window = cropAndRotate(image, stabilized_rect)
        else:
            view_window = self.__handleLostTarget()
        
        self.prev_window = view_window

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
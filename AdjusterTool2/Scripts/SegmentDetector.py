# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-03-20

import os
import cv2
import numpy as np

class SegmentDetector:
    def __init__(self, output_folder=None, segment_height=100, band_height=20, empty_frame_threshold=0.05):
        self.output_folder = output_folder
        self.segment_height = segment_height
        self.band_height = band_height
        self.empty_frame_threshold = empty_frame_threshold

        center_y = segment_height//2
        self.template_start_y = max(0, center_y - (band_height // 2))
        self.template_end_y = min(segment_height, center_y + (band_height // 2))
        
        self.total_displacement = 0
        self.segment_count = 0

        self.prev_image = None
        self.prev_mask = None
        self.prev_max_loc = None

    def __extractTemplate(self, image, mask):
        """
        Extracts a template from the image. 
        The template is a horizontal band from the center of the image.
        """

        # Extract the band from image and mask
        band = image[self.template_start_y:self.template_end_y, :]
        band_mask = mask[self.template_start_y:self.template_end_y, :]

        return band, band_mask

    def __templateMatching(self, image, template, template_mask):
        """
        Perform template matching based on the template and template_mask
        """

        # Perform template matching with the mask
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=template_mask)

        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if self.prev_max_loc is None:
            self.prev_max_loc = max_loc
        else:
            alpha = 0.7  # Smoothing factor (closer to 1 = more stable, but slower response)
            max_loc = (
                int(alpha * self.prev_max_loc[0] + (1 - alpha) * max_loc[0]),
                int(alpha * self.prev_max_loc[1] + (1 - alpha) * max_loc[1])
            )
            self.prev_max_loc = max_loc  # Store for next frame

        drawn_template = image.copy()

        # Draw a rectangle around the best match
        h, w = template.shape[:2]
        
        cv2.rectangle(drawn_template, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        cv2.line(drawn_template, (0, self.template_start_y), (image.shape[1], self.template_start_y), (0, 0, 255), 2)

        return drawn_template, max_loc

    def __checkEmptyFrame(self, mask):
        nonzero_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        nonzero_ratio = nonzero_pixels / total_pixels

        return nonzero_ratio < self.empty_frame_threshold        

    def trackDisplacement(self, image, mask):
        """
        Track the displacement from the last image to the next
        """
        
        if self.prev_image is None:
            self.prev_image = image
            self.prev_mask = mask
            return 0, image

        if self.__checkEmptyFrame(mask):
            self.prev_image = image
            self.prev_mask = mask
            return 0, image
        
        # Use template tracking to get new location
        template, template_mask = self.__extractTemplate(self.prev_image, self.prev_mask)
        drawn_template, max_loc = self.__templateMatching(image, template, template_mask)

        # Calculate displacement
        original_y, new_y = self.template_start_y, max_loc[1]
        displacement = original_y - new_y

        self.prev_image = image
        self.prev_mask = mask

        #print(displacement)
        #print(self.total_displacement)

        return displacement, drawn_template

    def checkNewSegment(self, image, mask):
        """
        Detects whether or not the image displays a unique segment

        Detection is based on tracking total vertical displacement. 
        Each segment is a fixed height. Declare a new segment is found
        every fixed increment of total displacement.
        """

        displacement, drawn_template = self.trackDisplacement(image, mask)
        self.total_displacement += displacement

        is_new_segment = self.total_displacement >= self.segment_count * self.segment_height

        return is_new_segment, drawn_template

    def detectSegment(self, image, mask):
        
        is_new_segment, drawn_template = self.checkNewSegment(image, mask)
        
        if is_new_segment:
            self.segment_count += 1
            print(f"Frame {self.segment_count} Detected!")
            if self.output_folder is not None:
                output_path = os.path.join(self.output_folder, f"frame_{self.segment_count}.jpg")
                cv2.imwrite(output_path, image)
        
        return drawn_template



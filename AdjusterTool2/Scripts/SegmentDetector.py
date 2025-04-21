# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18
# Last Modified: 2025-04-21

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Scripts.ResizeForDisplay import resize_for_display

class SegmentDetector:
    def __init__(self, output_folder=None, segment_height=100, band_height=40, empty_frame_threshold=0.02):
        self.output_folder = output_folder
        self.segment_height = segment_height
        self.band_height = band_height
        self.empty_frame_threshold = empty_frame_threshold

        center_y = segment_height//2
        self.template_start_y = max(0, center_y - (band_height // 2))
        self.template_end_y = min(segment_height, center_y + (band_height // 2))
        
        self.total_displacement = 0
        self.segment_count = 0
        self.frame_count = 0

        self.prev_image = None
        self.prev_mask = None
        self.prev_max_loc = None

    def restSegements(self):
        self.total_displacement = 0
        self.segment_count = 0
        self.frame_count = 0
        self.prev_image = None
        self.prev_mask = None
        self.prev_max_loc = None

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

    def __templateMatching(self, image, template):
        """
        Perform template matching based on the template
        """

        # Perform template matching with the mask
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        self.prev_max_loc = max_loc
        
        drawn_template = image.copy()
        drawn_template = cv2.cvtColor(drawn_template, cv2.COLOR_GRAY2RGB)

        # Draw a rectangle around the best match
        h, w = template.shape[:2]
        
        cv2.rectangle(drawn_template, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        cv2.line(drawn_template, (0, self.template_start_y), (image.shape[1], self.template_start_y), (0, 0, 255), 2)

        return drawn_template, max_loc

    def _checkEmptyFrame(self, mask):
        nonzero_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        nonzero_ratio = nonzero_pixels / total_pixels

        return nonzero_ratio < self.empty_frame_threshold        

    def trackDisplacement(self, image, mask):
        """
        Track the displacement from the last image to the next
        """

        preprocessed = self._imagePreprocessing(image)

        if self.frame_count % 1 == 0:  

            if self.prev_image is None:
                self.prev_image = preprocessed
                return 0, image

            if self._checkEmptyFrame(mask):
                self.prev_image = preprocessed
                return 0, image
            
            # Use template tracking to get new location
            template = self.__extractTemplate(self.prev_image)
            drawn_template, max_loc = self.__templateMatching(preprocessed, template)

            # Calculate displacement
            original_y, new_y = self.template_start_y, max_loc[1]
            displacement = abs(original_y - new_y)

            self.prev_image = preprocessed

            return displacement, drawn_template
        return 0, preprocessed

    def checkNewSegment(self, image, mask):
        """
        Detects whether or not the image displays a unique segment

        Detection is based on tracking total vertical displacement. 
        Each segment is a fixed height. Declare a new segment is found
        every fixed increment of total displacement.
        """

        displacement, drawn_template = self.trackDisplacement(image, mask)

        if displacement > 1:
            self.total_displacement += displacement
        self.frame_count += 1

        #print(f"displacement: {displacement}")
        #print(f"Total: {self.total_displacement}")

        # TODO: For some reason displacement is double counted, *2 correction
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
        
        return is_new_segment, drawn_template



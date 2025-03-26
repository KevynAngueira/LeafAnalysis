# Author: Kevyn Angueira Irizarry
# Created: 2025-03-26
# Last Modified: 2025-03-26

import os
import cv2
import numpy as np

from Scripts.SegmentDetector import SegmentDetector
from Scripts.ResizeForDisplay import resize_for_display

class PhaseCorrelationDetector(SegmentDetector):
    def __init__(self, output_folder=None, segment_height=100, empty_frame_threshold=0.05, step=5):
        self.output_folder = output_folder
        self.segment_height = segment_height
        self.empty_frame_threshold = empty_frame_threshold
        self.step = step

        self.total_displacement = 0
        self.segment_count = 0
        self.frame_count = 0

        self.prev_image = None
        self.prev_mask = None  

    def _imagePreprocessing(self, image, mask = None):
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

        # Mask -> Eliminate non-target areas
        #if mask is not None:
        #    gray = cv2.bitwise_and(gray, gray, mask=mask)

        # CLAHE -> Improve local contrast to boost subtle differences
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpen -> Further instensify differences
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        sharpened_float = sharpened.astype(np.float32) / 255.0

        cv2.imshow("Gray", resize_for_display(gray))
        cv2.imshow("CLAHE", resize_for_display(enhanced))
        cv2.imshow("Sharpened", resize_for_display(sharpened))
        cv2.imshow("Sharpened2", resize_for_display(sharpened_float))

        return sharpened_float

    def __phaseCorrelate(self, prev_proc, curr_proc):
        """
        Apply phase correlation to estimate shift (displacement) between the current and previous image
        """
        shift, response = cv2.phaseCorrelate(prev_proc, curr_proc)

        print(f"Phase correlation response: {response}")
        return shift

    def trackDisplacement(self, image, mask):
        """
        Track vertical displacement using phase correlation
        """
        
        if self.frame_count % self.step == 0:
            if self.prev_image is None:
                self.prev_image = image
                self.prev_mask = mask
                return 0

            if self._checkEmptyFrame(mask):
                self.prev_image = image
                self.prev_mask = mask
                return 0

            # Preprocess current and previous frames
            prev_proc = self._imagePreprocessing(self.prev_image, self.prev_mask)
            curr_proc = self._imagePreprocessing(image, mask)

            cv2.imshow("Prev Proc", resize_for_display(prev_proc))
            cv2.imshow("Curr Proc", resize_for_display(curr_proc))

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Get shift using phase correlation
            shift = self.__phaseCorrelate(prev_proc, curr_proc)

            displacement = abs(shift[1])  # Vertical shift

            self.prev_image = image
            self.prev_mask = mask

            return displacement
        return 0

    def checkNewSegment(self, image, mask):
        """
        Detects whether or not the image displays a unique segment.

        Detection is based on tracking total vertical displacement. 
        Each segment is a fixed height. Declare a new segment is found
        every fixed increment of total displacement.
        """
        
        displacement = self.trackDisplacement(image, mask)
        print(displacement)
        self.total_displacement += displacement
        self.frame_count += 1

        print(f"Displacement: {displacement}")
        print(f"Total: {self.total_displacement}")

        is_new_segment = self.total_displacement >= self.segment_count * self.segment_height
        return is_new_segment

    def detectSegment(self, image, mask):
        is_new_segment = self.checkNewSegment(image, mask)

        if is_new_segment:
            self.segment_count += 1
            print(f"Frame {self.segment_count} Detected!")

            if self.output_folder is not None:
                output_path = os.path.join(self.output_folder, f"frame_{self.segment_count}.jpg")
                cv2.imwrite(output_path, image)

        return is_new_segment



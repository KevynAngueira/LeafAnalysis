# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:41:12


import cv2
import numpy as np

from Scripts.ResizeForDisplay import resize_for_display

class ViewWindow:

    def imagePreprocessing(gray, blur=(5,5), morph_iterations=2, kernel_size=(10,10)):
        """
        Applying image preprocessing
            Morphological Close -> Closes small gaps
            Gaussian Blur -> Smoothens out edges
        """

        preprocessed = gray

        # Closes small gaps
        if morph_iterations > 0:
            kernel = np.ones(kernel_size, np.uint8)
            preprocessed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        preprocessed = cv2.GaussianBlur(preprocessed, blur, 0)

        cv2.imshow("Preprocessed", resize_for_display(preprocessed))

        return preprocessed

    def getContours(gray):
        """
        Get the contours on the image
        """

        edges = cv2.Canny(gray, 50, 150)

        # Find contours from Canny edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def drawContours(image, contours, color=(255,255,0)):
        """
        Draw contours on the given image
        """    
        def get_color(i):
            normalized = int(255 * (i % 10)/10 )
            return (255 - normalized, 0, normalized)  # Blue for small, Red for large

        drawn_contours = image.copy()
        
        if color is None:
            for i, c in enumerate(contours):
                color = get_color(i)
                cv2.drawContours(drawn_contours, [c], -1, color, 2)        
        else:
            cv2.drawContours(drawn_contours, contours, -1, color, 2)

        return drawn_contours

    def ViewWindowFromContours(contours, target_dimensions=(6.5, 1.0), tolerance=0.5):
        """
        Selects which contour represents the View Window based on the target dimension.
        
        View window is the largest contour matching the target_dimensions's aspect ratio,
        """

        target_rect = None
        target_box = None
        max_area = 0

        # Calculate the target aspect ratio
        target_aspect_ratio = target_dimensions[0]/target_dimensions[1]
        
        for contour in contours:
            # Get min area bounding box
            min_rect = cv2.minAreaRect(contour)
            w, h = min_rect[1]  

            aspect_ratio = max(w, h) / min(w, h) if min(w,h) != 0 else 0

            # Capture the largest contour that matches the target aspect ratio (within tolerance)
            if abs(aspect_ratio - target_aspect_ratio) <= tolerance:
                area = w*h
                if area > max_area:
                    # Get bbox coords
                    box = cv2.boxPoints(min_rect)
                    box = np.intp(box)
                    
                    # Update possible target
                    max_area = area
                    target_box = box
                    target_rect = min_rect
        
        return target_box, target_rect
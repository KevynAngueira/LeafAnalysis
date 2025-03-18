# Author: Kevyn Angueira Irizarry
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:41:12


import cv2
import numpy as np

from Scripts.HSVMask import HSVMask
from Scripts.ResizeForDisplay import resize_for_display

class LeafSeparator:
    def __init__(self,
        leaf_bounds=(np.array([0, 0, 0]), np.array([85, 255, 255])),
        front_base_bounds=(np.array([20, 0, 0]), np.array([168, 255, 255])),
        front_refinement_bounds=(np.array([0, 0, 0]), np.array([65, 255, 190])),
        target_dimensions=(6.5, 1.0)
    ):
        self.leaf_bounds = leaf_bounds
        self.front_base_bounds = front_base_bounds
        self.front_refinement_bounds = front_refinement_bounds

        self.target_dimensions = target_dimensions

        self.leafMask = HSVMask(leaf_bounds)
        self.frontBaseMask = HSVMask(front_base_bounds)
        self.frontRefinementMask = HSVMask(front_refinement_bounds)

    def __calculateFrontMask(self, image):
        """
        Calculating the front mask.
            front_mask = base_mask - refinement_mask

        Base mask loosely captures the frontpiece border. 
        Refinement mask removes non-frontpiece excess captured by the base mask
        """

        _, base_mask = self.frontBaseMask.applyHSVMask(image, True)
        _, refinement_mask = self.frontRefinementMask.applyHSVMask(image, True)
        
        front_mask = cv2.bitwise_and(base_mask, refinement_mask)

        return front_mask

    def __removeCenterContours(self, image, mask, border_threshold, blur, kernel_size, morph_iterations):
        """
        Refine the front mask by removing all the contours that are not touching the border.
        Removes all the excess contours not part of the frontpiece
        """

        # Remove Noise
        kernel = np.ones(kernel_size, np.uint8)
        morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

        blurred = cv2.GaussianBlur(morphed, blur, 0)

        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        refined_mask = blurred.copy()
        h, w = refined_mask.shape[:2]

        for contour in contours:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)

            # If contour is not within `border_threshold` pixels from any border, black it out
            if not (
                x < border_threshold or 
                y < border_threshold or 
                x + w_contour > w - border_threshold or 
                y + h_contour > h - border_threshold
            ):
                cv2.drawContours(refined_mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

        return refined_mask

    def getFrontMask(self, image, border_threshold=5, blur=(3,3), open_kernel_size=(3,3), close_kernel_size=(5,5), morph_iterations=2):
        """
        Gets the front mask
        """

        front_mask = self.__calculateFrontMask(image)
        border_front_mask = self.__removeCenterContours(image, front_mask, border_threshold, blur, open_kernel_size, morph_iterations)

        # Closes small gaps
        kernel = np.ones(close_kernel_size, np.uint8)
        refined_front_mask = cv2.morphologyEx(border_front_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        front_pixels = np.count_nonzero(refined_front_mask == 255)

        total_pixels = image.size
        white_pixel_percentage = (front_pixels / total_pixels) * 100

        print("Front Pixels and Percentage:")
        print(front_pixels)
        print(white_pixel_percentage)

        cv2.imshow("Front Mask", resize_for_display(front_mask))
        cv2.imshow("Border Mask", resize_for_display(border_front_mask))
        cv2.imshow("Refined Mask", resize_for_display(refined_front_mask))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return refined_front_mask, front_pixels

    def getLeafMask(self, image, front_mask):
        """
        Gets the leaf mask. Must remove the front mask to avoid non-leaf contours.
        """

        _, leaf_mask = self.leafMask.applyHSVMask(image)
        refined_leaf_mask = cv2.subtract(leaf_mask, front_mask)

        leaf_pixels = np.count_nonzero(refined_leaf_mask == 255)

        total_pixels = image.size
        white_pixel_percentage = (leaf_pixels / total_pixels) * 100

        print("Leaf Pixels and Percentage:")
        print(leaf_pixels)
        print(white_pixel_percentage)

        cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
        cv2.imshow("Front Mask", resize_for_display(front_mask))
        cv2.imshow("Refined Leaf Mask", resize_for_display(refined_leaf_mask))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return refined_leaf_mask, leaf_pixels

    def Extract(self, image):
        
        front_mask, front_pixels = self.getFrontMask(image)
        leaf_mask, leaf_pixels = self.getLeafMask(image, front_mask)

        total_pixels = image.size
        back_pixels = total_pixels-front_pixels

        leaf = cv2.bitwise_and(image, image, mask=leaf_mask)

        cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
        cv2.imshow("Leaf", resize_for_display(leaf))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return leaf, leaf_pixels, back_pixels
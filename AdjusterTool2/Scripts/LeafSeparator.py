import cv2
import numpy as np

from Scripts.HSVMask import HSVMask
from Scripts.ResizeForDisplay import resize_for_display

class LeafSeparator:
    def __init__(self,
        leaf_bounds=(np.array([75, 0, 0]), np.array([175, 255, 255])),
        target_dimensions=(650, 100)
    ):
        self.leaf_bounds = leaf_bounds
        self.target_dimensions = target_dimensions

        self.leafMask = HSVMask(leaf_bounds)
    
    def __crop_using_contours(self, image, border_margin=30, kernel_size=(3,3), morph_iterations=2, blur=(3,3)):
        """
        Crop the tool's frontpiece by detecting contours at the edges of the image and cropping to their mean height.
        """

        # Load the image (grayscale)
        height, width, _ = image.shape

        # Remove Noise
        kernel = np.ones(kernel_size, np.uint8)
        morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        blurred = cv2.GaussianBlur(morphed, blur, 0)

        gray = blurred[:, :, 0]

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect edge points
        left_points, right_points = [], []
        top_points, bottom_points = [], []

        for contour in contours:
            for point in contour[:, 0]:  # Extract (x, y) points
                x, y = point
                if x <= border_margin:
                    left_points.append(x)
                if x >= width - border_margin:
                    right_points.append(x)
                if y <= border_margin:
                    top_points.append(y)
                if y >= height - border_margin:
                    bottom_points.append(y)

        # Compute average values (fallback to original borders if no points found)
        left_bound = int(np.mean(left_points)) if left_points else 0
        right_bound = int(np.mean(right_points)) if right_points else width
        top_bound = int(np.mean(top_points)) if top_points else 0
        bottom_bound = int(np.mean(bottom_points)) if bottom_points else height

        # Crop the image
        cropped_image = image[top_bound:bottom_bound, left_bound:right_bound]

        return cropped_image

    def imagePreprocessing(self, image):
        """
        Applying preprocessing to the image
            Crop Frontpiece -> Crops out remaining frontpiece from view window
            Resize -> Resize image View Window to standardized size
        """

        cropped_image = self.__crop_using_contours(image)
        resized_image = cv2.resize(cropped_image, self.target_dimensions)

        return resized_image

    def Extract(self, image, display=False):
        """
        Extract the leaf only mask and count the leaf pixels
        """

        preprocessed = self.imagePreprocessing(image)

        leaf_result, leaf_mask = self.leafMask.applyHSVMask(preprocessed, True)
        leaf_pixels = np.count_nonzero(leaf_mask == 255)
        
        if display:
            cv2.imshow("Original", resize_for_display(image))
            cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
            cv2.imshow("Leaf Result", resize_for_display(leaf_result))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return leaf_result, leaf_pixels
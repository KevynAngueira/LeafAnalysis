# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-09-22

import cv2
import numpy as np

from LeafScan.Utils import resize_for_display

from LeafScan.Configs import LeafExtractorConfig

from LeafScan.Core.BinaryMask import LABMask

class LeafExtractor:
    def __init__(self, config: LeafExtractorConfig=None):
        if config is None:
            config = LeafExtractorConfig()
 
        self.__dict__.update(vars(config))

        self.leafMask = LABMask(config.leaf_bounds)

    def _crop_using_contours(self, image, **kwargs):
        """
        Crops the image to remove the tool's border using a color mask
        """

        height, width, _ = image.shape

        # Step 1: Apply tool color Mask
        a = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        # Compute Euclidean distance in the a-b space
        ab_distance = np.sqrt(a**2 + b**2)
        ab_distance_norm = cv2.normalize(ab_distance, None, 0, 255, cv2.NORM_MINMAX)
        tool_mask = ab_distance_norm.astype(np.uint8)

        # Step 2: Get color mask contours
        edges = cv2.Canny(tool_mask, 50, 150)
        cv2.imshow("Crop Edges", resize_for_display(edges))
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("⚠️ No contours found. Returning original image.")
            no_bounds = (0, height, 0, width)
            return image, no_bounds

        # Step 4: Combine all points
        all_points = np.vstack(contours).squeeze()

        # Step 5: Remove corner regions as they overlap with multiple border edges
        x, y = all_points[:, 0], all_points[:, 1]
        margin = self.border_margin

        not_in_top_left     = ~((x <= margin) & (y <= margin))
        not_in_top_right    = ~((x >= width - margin) & (y <= margin))
        not_in_bottom_left  = ~((x <= margin) & (y >= height - margin))
        not_in_bottom_right = ~((x >= width - margin) & (y >= height - margin))

        mask = not_in_top_left & not_in_top_right & not_in_bottom_left & not_in_bottom_right

        x = x[mask]
        y = y[mask]

        # Step 6: Remove regions likely to be part of the leaf
        x_center = int(np.mean(x))
        
        leaf_half_width = int(0.35 * width)
        x_min = max(0, x_center - leaf_half_width)
        x_max = min(width, x_center + leaf_half_width)

        x_min = max(x_min, margin)
        x_max = min(x_max, width - margin)

        not_in_leaf_strip = (x < x_min) | (x > x_max)

        top_vals = y[(y <= margin) & not_in_leaf_strip]
        bottom_vals = y[(y >= height - margin) & not_in_leaf_strip]

        # Step 7: Separate points into border edges
        left_vals   = x[x <= margin]
        right_vals  = x[x >= width - margin]
        top_vals    = y[(y <= margin) & not_in_leaf_strip]
        bottom_vals = y[(y >= height - margin) & not_in_leaf_strip]

        # Step 8: Capture the mean of each edge for future cropping
        left = min(int(np.max(left_vals)) if len(left_vals) > 0 else 0, margin)
        right = max(int(np.min(right_vals)) if len(right_vals) > 0 else width, width-margin)
        top = min(int(np.max(top_vals)) if len(top_vals) > 0 else 0, margin)
        bottom = max(int(np.min(bottom_vals)) if len(bottom_vals) > 0 else height, height-margin)

        # Step 9: Crop the original image to the new edge bounds, removin the tool border
        bounds = (top, bottom, left, right)
        cropped_image = image[top:bottom, left:right]
        
        return cropped_image, bounds

    def _imagePreprocessing(self, image, stabilize):
        """
        Applying preprocessing to the image
            Crop Frontpiece -> Crops out remaining frontpiece from view window
            Resize -> Resize image View Window to standardized size
        """

        #cropped_image, crop_bounds = self._crop_using_contours(image, stabilize=stabilize)

        resized_image = cv2.resize(image, self.target_dimensions)

        return resized_image #, crop_bounds

    def Extract(self, image, display=False, stabilize=True):
        """
        Extract the leaf-only mask, count leaf pixels, and calculate leaf percentage
        """

        preprocessed = self._imagePreprocessing(image, stabilize)

        leaf_result, leaf_mask = self.leafMask.applyMask(preprocessed, stabilize=stabilize)

        #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
        #if num_labels > 1:
        #    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        #    leaf_mask = np.uint8(labels == largest_label) * 255   
        #leaf_result = cv2.bitwise_and(preprocessed, preprocessed, mask=leaf_mask)
        
        if display:
                        
            #debug_img = image.copy()s
            #top, bottom, left, right = crop_bounds    
            #cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Original", resize_for_display(image))
            #cv2.imshow("Crop Box", debug_img)
            cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
            cv2.imshow("Leaf Result", resize_for_display(leaf_result))

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        return leaf_result, leaf_mask

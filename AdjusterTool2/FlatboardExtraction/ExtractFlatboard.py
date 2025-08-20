# Author: Kevyn Angueira Irizarry
# Created: 2025-06-25
# Last Modified: 2025-08-20

import os
import cv2
import numpy as np

import sys
from dataclasses import dataclass, field
from DetectLeaf import LeafDetector

sys.path.append("..")

from Scripts.ResizeForDisplay import resize_for_display
from Scripts.CropAndRotate import cropAndRotate
from Scripts.LABMask import LABMask

@dataclass
class FlatboardConfig:
    board_params: dict = field(default_factory=lambda: {
        "bounds": (np.array([0, 130, 0]), np.array([255, 255, 115])),
        #"bounds": (np.array([0, 0, 110]), np.array([255, 255, 255])),
        "size": (17.0, 32.0),
        "aspect_ratio": 17.0/32.0,
        "aspect_tolerance": 0.1
    })

    scale_params: dict = field(default_factory=lambda: {
        #"bounds": (np.array([125, 126, 122]), np.array([255, 255, 255])),
        "bounds": (np.array([0, 135, 105]), np.array([255, 255, 255])),
        "size": (5, 5),
        "aspect_ratio": 1,
        "aspect_tolerance": 0.2
    })

    kernel_size: tuple = (5, 5)
    morph_iterations: int = 2
    blur: tuple = (5, 5)


class FlatboardExtractor: 

    def __init__(self, flatboard_config: FlatboardConfig = None):
        if flatboard_config is None:
            flatboard_config = FlatboardConfig()
        
        self.__dict__.update(vars(flatboard_config))

        self.board_params["mask"] = LABMask(self.board_params["bounds"])
        self.scale_params["mask"] = LABMask(self.scale_params["bounds"])

        self.leafDetector = LeafDetector()

    def openImage(self, file_path):
        return cv2.imread(file_path)

    def displayImage(self, img, msg, wait=True):
        cv2.imshow(msg, resize_for_display(img))
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _imagePreprocessing(self, image, mask_obj):
        # Apply Mask
        _, preprocessed = mask_obj.applyMask(image)

        # Morphological Close
        #if self.morph_iterations > 0:
        #    kernel = np.ones(self.kernel_size, np.uint8)
        #    preprocessed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        # Gaussian Blur
        #preprocessed = cv2.GaussianBlur(preprocessed, self.blur, 0)     

        #self.displayImage(preprocessed, "Preprocessed", True)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(preprocessed, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            preprocessed = np.uint8(labels == largest_label) * 255   

        return preprocessed

    def _getContours(self, gray):
        """
        Get the contours on the image
        """

        #edges = cv2.Canny(gray, 50, 150)

        # Find contours from Canny edges
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def _drawContours(self, image, contours, color=(255,255,0)):

        image_shape = image.shape

        def get_color(i):
            normalized = int(255 * (i % 10)/10 )
            return (255 - normalized, normalized)

        drawn_contours = image.copy()

        if color is None:
            for i, contour in enumerate(contours):
                if contour is None or len(contour) < 4:
                    continue
                            
                draw_color = get_color(i) if color is None else color
                cv2.drawContours(drawn_contours, [contour], -1, draw_color, 2)
        else:
            cv2.drawContours(drawn_contours, contours, -1, color, 2)


        return drawn_contours

    def _contoursToTarget(self, contours, mask, params, filter_aspect=False, display=False):
        """
        Attempts to detect the view window from a list of contours
        (1) First tries finding a "direct match" (largest rect matching aspect ratio and surrounded by white)
        (2) Then tries fallback to scaled rect (eliminated protrusions, then largest rect surrounded by white)
        """

        target_box = None
        target_rect = None
        target_contour = None
        max_area = 0

        # Visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            if len(contour) < 4:
                continue  # Skip trivial or broken contours

            min_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(min_rect).astype(np.int32)
            w, h = min_rect[1]

            if w == 0 or h == 0: 
                continue # Skip trivial 2D contours

            area = w * h
            aspect_ratio = min(w, h) / max(w, h)
            if display: print(area, aspect_ratio)

            # Try direct match
            if filter_aspect and abs(aspect_ratio - params["aspect_ratio"]) > params["aspect_tolerance"]:
                continue
            
            #expanded_box = self.__expandRotatedBox(min_rect, padding=20)

            #if self.__isSurroundedByWhite(mask, expanded_box, box) and area > max_area:
            if area > max_area:
                target_box = box
                target_rect = min_rect
                target_contour = contour
                max_area = area
                    

            if display:
                cv2.drawContours(mask_vis, [box], -1, (0, 255, 0), 2)

        if display and target_box is not None:
            cv2.drawContours(mask_vis, [target_box], -1, (255, 0, 0), 2)
            cv2.imshow("Mask Vis", resize_for_display(mask_vis))
            cv2.waitKey(0)

        return target_rect, target_contour

    def _isolateTarget(self, image, target_contour):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [target_contour], -1, 255, -1)
        isolated = cv2.bitwise_and(image, image, mask=mask)
        return isolated

    def processImage(self, original_image, params, is_width_greater=False, display=False):

        mask_obj = params["mask"]

        preprocessed = self._imagePreprocessing(original_image, mask_obj)
        
        self.displayImage(preprocessed, "Preprocessed", True)

        contours = self._getContours(preprocessed)

        target_rect, target_contour = self._contoursToTarget(contours, preprocessed, params, display=display)

        isolated = self._isolateTarget(original_image, target_contour)

        self.displayImage(isolated, "Isolated", True)

        target = cropAndRotate(isolated, target_rect, is_width_greater)

        return target_rect, target, target_contour

    def _scaleObjectToPixelScale(self, scale_rect, scale_size, scale_contour):
        
        contour_area = cv2.contourArea(scale_contour)
        
        #pixel_w, pixel_h = scale_rect[1]
        #pixel_area = pixel_w * pixel_h

        true_w, true_h = scale_size
        true_area = true_w * true_h

        # Multiply any new pixel area by pixel scale to transform into true area
        #pixel_scale = true_area / pixel_area

        contour_scale = true_area / contour_area

        #print(pixel_w, pixel_h)
        #print(contour_area)
        #print(pixel_scale)
        print(contour_scale)

        return contour_scale

    def Extract(self, image_path, display=False, output_path=None):
        original_image = self.openImage(image_path)

        # Contours to Board
        board_rect, board, _ = self.processImage(original_image, self.board_params, display=False)

        # Contours to Pixel Scale
        scale_rect, scale, scale_contour = self.processImage(board, self.scale_params, True, display=True)
        pixel_scale = self._scaleObjectToPixelScale(scale_rect, self.scale_params['size'], scale_contour)

        board_result, board_mask = self.board_params['mask'].applyMask(board, invert_range=True)
        board_contours = self._getContours(board_mask)
       
        stacked_image, leaf_segments = self.leafDetector.detect_leaf_segments(board_result, board_contours)
        leaf_areas = [ cv2.contourArea(cnt) * pixel_scale for leaf_img, leaf_mask, cnt in leaf_segments ]
        leaf_total_area = sum(leaf_areas)

        print("====== Segments ======")
        print(leaf_areas)
        print(leaf_total_area)
            
        if display:
            self.displayImage(original_image, "Original Image", False)
            self.displayImage(board, "Board", False)
            self.displayImage(scale, "Scale")  
            
            drawn_contours = self._drawContours(board, board_contours)
            self.displayImage(board_mask, "Board Mask", False)
            self.displayImage(drawn_contours, "Drawn Contours", False)
            self.displayImage(board_result, "Board Result")  

            for idx, leaf_segment in enumerate(leaf_segments):
                leaf_img, leaf_mask, cnt = leaf_segment               
                self.displayImage(leaf_mask, f"Contour_{idx}", False)
                self.displayImage(leaf_img, f"Segment_{idx}", True) 

            self.displayImage(stacked_image, "Leaf Result") 

        if output_path is not None:
            cv2.imwrite(output_path, stacked_image)

        return stacked_image, leaf_total_area, leaf_areas


if __name__ == "__main__":
    flatboard = FlatboardExtractor()

    IMG_BASE_PATH = "/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Private/LeafScan-CornDefoliation2025-v1/data/field_02/plant_00/leaf_08/real"
    OUT_BASE_PATH = "/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Private/LeafScan-CornDefoliation2025-v1/data/field_02/plant_00/leaf_08/output"

    id_base = "02_00_08_00"

    IMG_PATH = f"{IMG_BASE_PATH}/img_{id_base}.jpg"
    OUT_PATH = f"{OUT_BASE_PATH}/out_{id_base}.jpg"
    
    flatboard.Extract(IMG_PATH, True, OUT_PATH)

# Author: Kevyn Angueira Irizarry
# Created: 2025-06-25
# Last Modified: 2025-09-22

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
        "bounds": (np.array([0, 135, 0]), np.array([255, 255, 150])),
        #"bounds": (np.array([0, 0, 110]), np.array([255, 255, 255])),
        "size": (25.0, 34.0),
        "aspect_ratio": 25.0/34.0,
        "aspect_tolerance": 0.1
    })

    scale_params: dict = field(default_factory=lambda: {
        "bounds": (np.array([0, 0, 0]), np.array([255, 130, 255])),
        
        #"bounds": (np.array([125, 126, 122]), np.array([255, 255, 255])),
        #"bounds": (np.array([0, 0, 0]), np.array([255, 145, 255])),
        "size": (5, 5),
        "aspect_ratio": 1,
        "aspect_tolerance": 0.1
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

        #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(preprocessed, connectivity=8)
        #if num_labels > 1:
        #    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        #    preprocessed = np.uint8(labels == largest_label) * 255   

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

        w, h = target_rect[1]

        return target_rect, target_contour

    def _isolateTarget(self, image, target_contour):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [target_contour], -1, 255, -1)
        isolated = cv2.bitwise_and(image, image, mask=mask)
        return isolated

    def _orderPoints(self, pts):
        # pts: (4,2) float32
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _approxQuad(self, contour, fallback_rect=None):
        """Try to approximate a 4-point polygon from a contour.
        If it fails, fall back to minAreaRect box (still 4 points)."""
        peri = cv2.arcLength(contour, True)
        # Try a few epsilons from tight to loose
        for frac in [0.015, 0.02, 0.03, 0.05, 0.08, 0.12]:
            approx = cv2.approxPolyDP(contour, frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        # Fallback
        if fallback_rect is None:
            fallback_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(fallback_rect).astype(np.float32)
        return box

    def _homographyFromSquare(self, src_quad, square_size_in=(5.0, 5.0), ppi=200):
        """Build H that maps the detected square (src_quad) to a metric plane where
        the square is exactly 5″×5″ at the given PPI."""
        src = self._orderPoints(src_quad.astype(np.float32))
        W_in, H_in = square_size_in
        dst_W = int(round(W_in * ppi))
        dst_H = int(round(H_in * ppi))
        dst = np.array([[0,0],[dst_W-1,0],[dst_W-1,dst_H-1],[0,dst_H-1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return H, (dst_W, dst_H), ppi

    def _warpWithPadForBoard(self, image, H, board_quad, pad = 500):
        """Warp board onto a canvas with fixed padding around all sides."""
        board_quad = self._orderPoints(board_quad.astype(np.float32))
        board_quad_h = cv2.perspectiveTransform(board_quad.reshape(1,4,2), H)[0]
        xs, ys = board_quad_h[:,0], board_quad_h[:,1]

        min_x, min_y = float(np.min(xs)), float(np.min(ys))
        max_x, max_y = float(np.max(xs)), float(np.max(ys))

        out_W = int(np.ceil(max_x - min_x + pad))
        out_H = int(np.ceil(max_y - min_y + pad))

        # Shift homography to place object in middle with padding
        T = np.array([[1, 0, pad - min_x],
                    [0, 1, pad - min_y],
                    [0, 0, 1]], dtype=np.float32)
        H_shift = T @ H

        warped = cv2.warpPerspective(image, H_shift, (out_W, out_H))
        return warped, H_shift, (out_W, out_H)


    def processImage(self, original_image, params, is_width_greater=False, display=False):

        mask_obj = params["mask"]

        preprocessed = self._imagePreprocessing(original_image, mask_obj)
        
        #self.displayImage(preprocessed, "Preprocessed", True)

        contours = self._getContours(preprocessed)

        target_rect, target_contour = self._contoursToTarget(contours, preprocessed, params, True, display=display)

        quad = self._approxQuad(target_contour, fallback_rect=target_rect)
        quad = self._orderPoints(quad)

        isolated = self._isolateTarget(original_image, target_contour)

        #self.displayImage(isolated, "Isolated", True)

        target = cropAndRotate(isolated, target_rect, is_width_greater)

        return target_rect, target, target_contour, quad

    def Extract(self, image_path, display=False, output_path=None, ppi=100):
        original_image = self.openImage(image_path)

        # Contours to Board
        board_rect, board_img, board_contour, board_quad = self.processImage(original_image, self.board_params, display=False)

        # Contours to Pixel Scale
        scale_rect, scale_img, scale_contour, scale_quad = self.processImage(board_img, self.scale_params, True, display=False)
        
        # Homography from scale
        
        H_sq, (sq_W, sq_H), px_per_inch = self._homographyFromSquare(scale_quad, square_size_in=self.scale_params['size'], ppi=ppi)

        # Rectify board and scale based on homography (top-down, undistorted)
        rectified_board, H_board, (out_W, out_H) = self._warpWithPadForBoard(board_img, H_sq, board_quad)

        # Segment leaf sections on rectified board
        board_result, board_mask = self.board_params['mask'].applyMask(rectified_board, invert_range=True)
        board_contours = self._getContours(board_mask)

        # Stack leaf sections
        stacked_image, leaf_segments = self.leafDetector.detect_leaf_segments(board_result, board_contours)

        # Calc scale reference
        inch_per_px = 1.0 / float(px_per_inch)
        px_to_in2 = inch_per_px ** 2

        # Calc total leaf area
        leaf_areas_in2 = [ cv2.contourArea(cnt) * px_to_in2 for _, _, cnt in leaf_segments ]
        leaf_total_area_in2 = float(sum(leaf_areas_in2))


        # Debug
        print("PPI (px per inch):", px_per_inch)
        print("Leaf areas (in^2):", leaf_areas_in2)
        print("Leaf total area (in^2):", leaf_total_area_in2)
            
        if display:
            self.displayImage(original_image, "Original Image", False)
            self.displayImage(scale_img, "Scale")

            board_side_by_side = cv2.hconcat([
                cv2.resize(board_img, (rectified_board.shape[1], rectified_board.shape[0])),
                rectified_board
            ])
            self.displayImage(board_side_by_side, "Board: Original vs Rectified")

            drawn_contours = self._drawContours(rectified_board, board_contours)
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

        return stacked_image, leaf_total_area_in2, leaf_areas_in2


if __name__ == "__main__":
    flatboard = FlatboardExtractor()

    fid = "07"
    pid = "06"
    lid = "21"
    mid = "01"

    subtype = "s"
    status = "simulated"
    
    BASE_PATH = f"/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Private/LeafScan-CornDefoliation2025-v1/data/field_{fid}/plant_{pid}/leaf_{lid}/{status}"

    id_base = f"{fid}_{pid}_{lid}_{mid}"

    IMG_PATH = f"{BASE_PATH}/media/img_{subtype}m_{id_base}.jpg"
    OUT_PATH = f"{BASE_PATH}/output/img_{subtype}m_{id_base}.jpg"

    flatboard.Extract(IMG_PATH, True, OUT_PATH)

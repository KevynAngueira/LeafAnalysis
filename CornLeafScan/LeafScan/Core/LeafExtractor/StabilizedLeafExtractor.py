# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-24

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from LeafScan.Utils import resize_for_display

from LeafScan.Configs import LeafExtractorConfig

from LeafScan.Core.BinaryMask import LABMask

from .KmeansLeafExtractor import KmeansLeafExtractor

class StabilizedLeafExtractor(KmeansLeafExtractor):
    def __init__(self, config: LeafExtractorConfig=None, alpha: float = 0.2, threshold: int = 200):
        super().__init__(config)

        self.prev_ema_mask = None

        self.alpha = alpha 
        self.threshold = threshold

    def reset(self):
        self.prev_leaf = None
        self.current_leaf = None

    def applyStabilization(self, new_frame, new_mask):
        current_ema_mask = new_mask.astype(np.float32)
        
        if self.prev_ema_mask is None:
            self.prev_ema_mask = current_ema_mask.copy()
            return new_frame, new_mask

        self.prev_ema_mask[:] = (self.alpha * current_ema_mask) + ((1 - self.alpha) * self.prev_ema_mask)
        stabilized_mask = (self.prev_ema_mask >= self.threshold).astype(np.uint8) * 255
        stabilized_frame = cv2.bitwise_and(new_frame, new_frame, mask=stabilized_mask)
        
        return stabilized_frame, stabilized_mask

    def Extract(self, image, display=False, deep_display=False, stabilize=True):
        leaf_result, leaf_mask = super().Extract(image, display, deep_display)

        if stabilize:
            stabilized_result, stabilized_mask = self.applyStabilization(leaf_result, leaf_mask)
        else:
            stabilized_result, stabilized_mask = leaf_result, leaf_mask

        if display:
            cv2.imshow("Stabilized", resize_for_display(stabilized_result))
            cv2.imshow("Difference", resize_for_display(cv2.absdiff(leaf_mask, stabilized_mask)))
        
        return stabilized_result, stabilized_mask

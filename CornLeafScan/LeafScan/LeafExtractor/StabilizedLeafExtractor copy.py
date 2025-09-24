# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Misc.ResizeForDisplay import resize_for_display

from Configs import LeafExtractorConfig

from BinaryMask import LABMask

from .ContourLeafExtractor import ContourLeafExtractor

class StabilizedLeafExtractor(ContourLeafExtractor):
    def __init__(self, config: LeafExtractorConfig=None, alpha: float = 0.1, threshold: int = 200):
        super().__init__(config)

        self.prev_ema_mask = None

        self.alpha = alpha 
        self.threshold = threshold

    def reset(self):
        self.prev_leaf = None
        self.current_leaf = None

    def applyStabilization(self, new_frame):
        current_ema_mask = (new_frame > 0).astype(np.float32)
        
        if self.prev_ema_mask is None:
            return new_frame, current_ema_mask

        self.prev_ema_mask[:] = (self.alpha * current_ema_mask) + ((1 - self.alpha) * self.prev_ema_mask)
        stabilized_mask = (self.prev_ema_mask >= self.threshold).astype(np.uint8) * 255
        stabilized_frame = cv2.bitwise_and(new_frame, stabilized_mask)
        
        return stabilized_frame, stabilized_mask

    def Extract(self, image, display=False, deep_display=False, stabilize=True):
        leaf_result, leaf_mask = super().Extract(image, display, deep_display)

        if stabilize:
            leaf_result, leaf_mask = self.applyStabilization(leaf_result)
        
        return leaf_result, leaf_mask

# Author: Your Name
# Created: 2025-03-18 13:41:12
# Last Modified: 2025-03-18 13:43:18


import cv2
import numpy as np

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig


class StabilizedWindowSeparator(ViewWindow):
    def __init__(self, config: ViewWindowConfig = None, alpha: float = 0.1):
        super().__init__(config)
        
        self.prev_center = None
        self.frame_skip_count = 0 
        self.alpha = alpha

    def __calculateCenter(self, rect):
        

# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-09-22

__all__ = ['ViewExtractorConfig', 'LeafExtractorConfig']

import numpy as np
from dataclasses import dataclass

@dataclass
class ViewExtractorConfig:
    tool_bounds: tuple = (np.array([165, 130, 85]), np.array([255, 170, 255]))
    target_aspect_ratio: float = 6.5
    aspect_ratio_tolerance: float = 0.8
    kernel_size: tuple = (5, 5)
    morph_iterations: int = 2
    blur: tuple = (5, 5)
    padding: int = 20

@dataclass
class LeafExtractorConfig:
    leaf_bounds: tuple = (np.array([0, 0, 124]), np.array([255, 135, 255]))
    target_dimensions: tuple = (650, 100)
    
    morph_kernel: tuple = (3, 3)
    blur_kernel: tuple = (5, 5)
    dilate_kernel: tuple = (3, 3)

    morph_iterations: int = 2
    dilate_iterations: int = 2
    
    sharpen_weight: tuple = (2, -0.5)
    canny_coef: tuple = (0.5, 1.5)

    kmeans_clusters: int = 3
    ellipse_kernel: tuple = (3,3)
    padding: int = 20
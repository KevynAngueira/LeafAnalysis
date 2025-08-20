# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

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

@dataclass
class LeafExtractorConfig:
    #leaf_bounds: tuple = (np.array([0, 0, 100]), np.array([255, 135, 255]))
    leaf_bounds: tuple = (np.array([0, 0, 110]), np.array([255, 255, 255]))
    target_dimensions: tuple = (650, 100)
    border_margin: int = 10
    kernel_size: tuple = (3, 3)
    morph_iterations: int = 2
    blur: tuple = (3, 3)

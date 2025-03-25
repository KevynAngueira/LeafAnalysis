# Author: Kevyn Angueira Irizarry
# Created: 2025-03-25
# Last Modified: 2025-03-25



import cv2
import numpy as np


class LeafAreaCalculator:
    def __init__(self, window_dimensions = (6.5, 1.0)):
        self.segment_area = window_dimensions[0]*window_dimensions[1]
        print("Segment Area")
        print(self.segment_area)
        
        self.leaf_area_arr = []
        self.rolling_leaf_area = 0

    def calculateSegment(self, leaf_mask):
        leaf_pixels = np.count_nonzero(leaf_mask == 255)
        total_pixels = leaf_mask.size
        leaf_percentage = (leaf_pixels / total_pixels)

        leaf_area = leaf_percentage * self.segment_area
    
        self.leaf_area_arr.append(leaf_area)

        self.rolling_leaf_area += leaf_area

        return leaf_area

    def getTotalArea(self):
        return self.rolling_leaf_area

    def getAllAreas(self):
        return self.leaf_area_arr


    

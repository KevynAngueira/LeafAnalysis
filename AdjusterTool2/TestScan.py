# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-03-19


import os
import cv2
import numpy as np

from Scripts.LeafScan import LeafScan

video_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/vid1.mp4"
output_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/Results/vid1.mp4"

segement_folder = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/LeafSegments"

leafScan = LeafScan(output_folder=segement_folder)
leafScan.scanVideo(video_path, output_path)

# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-03-25


import os
import cv2
import numpy as np

from Scripts.LeafScan import LeafScan

video_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/leaf1-1.mp4"
output_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/frames"

segement_folder = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/LeafSegments"

os.system(f"rm -rf {segement_folder}/*")

leafScan = LeafScan(output_folder=segement_folder)
leafScan.scanVideo(video_path, output_path)

# Author: Your Name
# Created: 2025-03-18 13:34:11
# Last Modified: 2025-03-18 13:34:11
import os
import cv2
import numpy as np

from Scripts.LeafScan import LeafScan

video_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/vid2.mp4"
output_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/Results"

leafScan = LeafScan()
leafScan.scanVideo(video_path, output_path)

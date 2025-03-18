import os
import cv2
import numpy as np

from Scripts.LeafScan import LeafScan

video_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/vid3.mp4"
output_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/Results"

leafScan = LeafScan()
leafScan.scanVideo(video_path, output_path)

import os
import shutil
import numpy as np

from ProcessVideo import VideoDetection
from TemplateTracker import TemplateTracker, process_video
from DisplaySegments import stitch_images

leaf_areas = []
base_video_name = "demonstration"
num_videos = 4

for i in range(num_videos):
    video_name = f"{base_video_name}{i}"
    video_path = f"TestVideos/{video_name}.mp4"
    video_folder = f"TestVideos/{video_name}"
    
    leaf_path = f"{video_folder}/leaf_output.mp4"
    segments_folder = f"LeafSegments/{video_name}"

    # Remove the folder if it exists
    if os.path.exists(segments_folder):
        shutil.rmtree(segments_folder)
    os.makedirs(video_folder, exist_ok=True) 
    os.makedirs(segments_folder, exist_ok=True) 

    # == Process Video ==

    # Define tool and leaf adaptive hsv mask
    orange_hsv_thresholds = (np.array([2, 115, -1]), np.array([12, 255, 255]))
    leaf_hsv_thresholds = (np.array([35, 23, 0]), np.array([100, 255, 215]))
    # Define target area dimensions
    target_dimensions = (6.5, 1)

    videoDetection = VideoDetection(orange_hsv_thresholds, leaf_hsv_thresholds, target_dimensions)
    videoDetection.process_video(video_path, leaf_path)

    # == Template Tracker ==
    process_video(leaf_path, segments_folder)

    # == Display Segments == 
    total_area = stitch_images(segments_folder, video_folder)
    leaf_areas.append(total_area)

print("Leaf Areas:")
print(leaf_areas)
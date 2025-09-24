# Author: Kevyn Angueira Irizarry
# Created: 2025-09-24
# Last Modified: 2025-09-24

import json
import shutil
import numpy as np
from pathlib import Path

from LeafScan import LeafScan
from LeafScan.Configs import ViewExtractorConfig, LeafExtractorConfig

from LeafScan.Utils import select_video_main, stitch_images_vertically

def noisy_round(value, precision):
    noise = np.random.uniform(-precision, precision)
    return round((value + noise) / precision) * precision

def run_scan():
    print("Select Video:")
    print("(0) vid_hm_07_06_12_00")
    print("(1) vid_sm_07_06_20_00")
    choice = input()

    if choice == 0:
        video_path = Path("Test_Data/vid_hm_07_06_12_00.mp4")
        remaining_length = 34.125
    else:
        video_path = Path("Test_Data/vid_sm_07_06_20_00.mp4")
        remaining_length = 25.25


    slices_folder = Path("tmp_slices")
    if slices_folder.exists():
        shutil.rmtree(slices_folder)
    slices_folder.mkdir(parents=True, exist_ok=True)

    # Step 4: Run LeafScan
    print(f"▶️ Running LeafScan on video: {video_path}")
    print(f"📏 Using rough remaining length: {remaining_length} inches")
    print(f"📁 Output folder: {slices_folder}")

    view_config = ViewExtractorConfig()
    separator_config = LeafExtractorConfig()

    view_config.tool_bounds = (np.array([0, 145, 0]), np.array([255, 255, 255]))
    
    
    leaf_scan = LeafScan(
        view_config=view_config,
        leaf_config=separator_config,
        output_folder=slices_folder, 
        display=True,
        deep_display=True
    )
    
    output_filename = video_path.name
    video_output_path = slices_folder / output_filename
    stacked_slices_path = video_output_path.with_suffix('.jpg')

    scanned_area = leaf_scan.scanVideo(
        remaining_leaf_length=remaining_length,
        video_path=str(video_path),
        output_path=str(video_output_path)
    )

    stitch_images_vertically(slices_folder, stacked_slices_path)

    print(f"✅ Scanned Remaining Area: {scanned_area:.2f} square units")

if __name__ == "__main__":
    run_scan()

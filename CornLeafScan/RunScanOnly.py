# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import shutil
import numpy as np
from pathlib import Path

from Scripts.LeafScan import LeafScan
from Scripts.ViewWindow import ViewWindowConfig
from Scripts.LeafSeparator import LeafSeparatorConfig

from SelectLeafVideo import main as select_video_main

def noisy_round(value, precision):
    noise = np.random.uniform(-precision, precision)
    return round((value + noise) / precision) * precision

def run_scan():
    # Step 1: Get selected video path from selector script
    video_path = select_video_main()
    if video_path is None:
        print("❌ Video selection failed.")
        return

    # Step 2: Define output folder
    output_folder = Path("tmp_output")
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Step 3: Set hardcoded remaining length and apply rounding error
    true_remaining_length = 34  # hardcoded value in inches
    rounding_precision = 1 / 8
    rough_remaining_length = noisy_round(true_remaining_length, rounding_precision)

    # Step 4: Run LeafScan
    print(f"
▶️ Running LeafScan on video: {video_path}")
    print(f"📏 Using rough remaining length: {rough_remaining_length} inches")
    print(f"📁 Output folder: {output_folder}")

    view_config = ViewWindowConfig()
    separator_config = LeafSeparatorConfig()

    view_config.tool_bounds = (np.array([60, 130, 0]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 150, 110]), np.array([255, 255, 255]))

    separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 130, 255]))

    leaf_scan = LeafScan(
        view_config=view_config,
        leaf_config=separator_config,
        output_folder=output_folder, 
        display=True
    )
    
    # Note: If rough_base_widths is not needed, omit it or pass None
    scanned_area = leaf_scan.scanVideo(
        remaining_leaf_length=rough_remaining_length,
        video_path=str(video_path),
        output_path=str(output_folder / "scan_output.mp4")
    )

    print(f"
✅ Scanned Remaining Area: {scanned_area:.2f} square units")

if __name__ == "__main__":
    run_scan()

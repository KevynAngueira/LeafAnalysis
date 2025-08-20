# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import json
import shutil
import numpy as np
from pathlib import Path

from LeafScan import LeafScan
from Configs import ViewExtractorConfig, LeafExtractorConfig

from Misc.SelectLeafVideo import main as select_video_main
from Misc.StackImages import stitch_images_vertically

def noisy_round(value, precision):
    noise = np.random.uniform(-precision, precision)
    return round((value + noise) / precision) * precision

def get_current_leaf_length(video_path: Path) -> float:
    """
    Parses field, plant, and leaf IDs from video path and loads corresponding leaf_metadata.json.
    """
    # Example: .../field_02/plant_00/leaf_07/defoliated/media/vid_dm_02_00_07_001.mp4
    parts = video_path.parts

    leaf_id = next(part.split("_")[1] for part in parts if part.startswith("leaf_"))

    metadata_path = video_path.parents[3] / "leaf_metadata.json"  # 3 levels up from media
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Could not find metadata at {metadata_path}")

    with open(metadata_path, "r") as f:
        leaf_metadata = json.load(f)

    leaf_key = f"leaf_{leaf_id}"
    try:
        length_str = leaf_metadata[leaf_key]["leaf_description"]["measurements"]["max_length"]["current"]
        return float(length_str)
    except KeyError:
        raise ValueError(f"❌ Could not extract current max_length from {metadata_path}")

def run_scan():
    # Step 1: Get selected video path from selector script
    video_path = select_video_main()
    if video_path is None:
        print("❌ Video selection failed.")
        return

    # Step 2: Define output folder
    slices_folder = Path("tmp_slices")
    if slices_folder.exists():
        shutil.rmtree(slices_folder)
    slices_folder.mkdir(parents=True, exist_ok=True)

    media_folder = video_path.parent  # .../media/
    status_folder = media_folder.parent  # .../defoliated/
    out_folder = status_folder / "output"

    # Step 3: Set hardcoded remaining length and apply rounding error
    true_remaining_length = get_current_leaf_length(video_path)  # hardcoded value in inches
    rounding_precision = 1 / 8
    #rough_remaining_length = noisy_round(true_remaining_length, rounding_precision)
    rough_remaining_length = true_remaining_length

    # Step 4: Run LeafScan
    print(f"
▶️ Running LeafScan on video: {video_path}")
    print(f"📏 Using rough remaining length: {rough_remaining_length} inches")
    print(f"📁 Output folder: {slices_folder}")

    view_config = ViewExtractorConfig()
    separator_config = LeafExtractorConfig()

    view_config.tool_bounds = (np.array([100, 138, 115]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 145, 110]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 145, 105]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 145, 115]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 138, 115]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 138, 110]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 138, 120]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 138, 124]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([0, 140, 115]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([50, 138, 105]), np.array([255, 255, 255]))
    #view_config.tool_bounds = (np.array([50, 137, 117]), np.array([255, 255, 255]))
    

    separator_config.leaf_bounds = (np.array([0, 0, 126]), np.array([255, 130, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 134, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 126]), np.array([255, 130, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 110]), np.array([255, 130, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 128]), np.array([255, 140, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 125]), np.array([255, 136, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 130, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 125]), np.array([255, 145, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 132, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 125]), np.array([255, 132, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 128]), np.array([255, 140, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 110]), np.array([255, 128, 255]))    
    #separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 130, 255]))
    #separator_config.leaf_bounds = (np.array([0, 0, 120]), np.array([255, 135, 255]))

    leaf_scan = LeafScan(
        view_config=view_config,
        leaf_config=separator_config,
        output_folder=slices_folder, 
        display=True,
        deep_display=True
    )
    

    output_filename = video_path.name
    video_output_path = out_folder / output_filename
    stacked_slices_path = video_output_path.with_suffix('.jpg')

    scanned_area = leaf_scan.scanVideo(
        remaining_leaf_length=rough_remaining_length,
        video_path=str(video_path),
        output_path=str(video_output_path)
    )

    stitch_images_vertically(slices_folder, stacked_slices_path)

    print(f"
✅ Scanned Remaining Area: {scanned_area:.2f} square units")

if __name__ == "__main__":
    run_scan()

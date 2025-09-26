# Author: Kevyn Angueira Irizarry
# Created: 2025-09-24
# Last Modified: 2025-09-26

import json
import joblib
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from LeafScan import LeafScan
from LeafScan.Configs import ViewExtractorConfig, LeafExtractorConfig
from LeafScan.Utils import stitch_images_vertically
from LeafScan.Utils.GetLeafRecord import GetLeafRecord

# === Config ===
MODEL_PATH = "LeafScan/Models/gradient_boosting_model.pkl"
RESULTS_FILE = "Results/batch_defoliation_results.csv"
PROGRESS_FILE = "Resultsprogress.txt"

BASE_DIR = Path("/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Private/LeafScan-CornDefoliation2025-v1/data/")

# === Utilities ===
def noisy_round(value, precision):
    noise = np.random.uniform(-precision, precision, size=np.shape(value))
    return np.round((value + noise) / precision) * precision

def get_current_leaf_length(video_path: Path, status: str) -> float:
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
        if status == "healthy":
            subfolder = "original"
        else:
            subfolder = "simulated"

        length_str = leaf_metadata[leaf_key]["leaf_description"]["measurements"]["max_length"][subfolder]
        return float(length_str)
    except KeyError:
        raise ValueError(f"❌ Could not extract max_length from {metadata_path}")

def get_all_defoliated_videos():
    """Traverse dataset for all defoliated videos."""
    return sorted(BASE_DIR.glob("field_*/plant_*/leaf_*/simulated/media/*.mp4"))

def prompt_resume_options():
    print("🔁 Previous progress detected.")
    print("1: Resume from where left off")
    print("2: Restart from scratch")
    print("3: Start from custom index")
    while True:
        try:
            choice = int(input("Select an option: "))
            if choice in [1, 2, 3]:
                return choice
        except ValueError:
            pass
        print("Invalid choice. Try again.")

def load_progress():
    if Path(PROGRESS_FILE).exists():
        return int(Path(PROGRESS_FILE).read_text().strip())
    return 0

def save_progress(index):
    Path(PROGRESS_FILE).write_text(str(index))

def append_result(row):
    header = not Path(RESULTS_FILE).exists()
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode='a', index=False, header=header)

# === Main pipeline ===
def main():
    # Load model
    model = joblib.load(MODEL_PATH)

    # Get all videos
    videos = get_all_defoliated_videos()
    if not videos:
        print("❌ No defoliated videos found.")
        return

    start_index = 0
    if Path(PROGRESS_FILE).exists():
        choice = prompt_resume_options()
        if choice == 1:  # Resume
            start_index = load_progress()
        elif choice == 2:  # Restart
            save_progress(0)
            start_index = 0
        elif choice == 3:  # Custom index
            while True:
                try:
                    start_index = int(input(f"Enter custom start index (0–{len(videos)-1}): "))
                    if 0 <= start_index < len(videos):
                        break
                except ValueError:
                    pass
                print("Invalid index. Try again.")

    print(f"▶️ Resuming at index {start_index}")

    for i, video_path in enumerate(videos[start_index:], start=start_index):
        try:
            print(f"\n📹 [{i+1}/{len(videos)}] Processing {video_path}")

            media_folder = video_path.parent  # .../media/
            status_folder = media_folder.parent  # .../simulated/
            out_folder = status_folder / "output"
            out_folder.mkdir(parents=True, exist_ok=True)

            # Parse entry IDs from filename
            # Example: vid_sm_02_00_07_001.mp4 → ["vid", "sm", "02", "00", "07", "001"]
            parts = video_path.stem.split("_")
            _, _, f_id, p_id, l_id, m_id = parts

            # Get entry record
            params, metrics = GetLeafRecord(f_id, p_id, l_id)
            true_original_area, true_remaining_area, true_defoliation = metrics

            # Estimate original area
            pred_original_area = model.predict(params)[0]

            # Get current leaf length
            true_remaining_length = get_current_leaf_length(video_path, status_folder.name)
            rough_remaining_length = noisy_round(true_remaining_length, 1/8)

            # Run LeafScan
            view_config = ViewExtractorConfig()
            separator_config = LeafExtractorConfig()
            view_config.tool_bounds = (np.array([0, 145, 0]), np.array([255, 255, 255]))

            slices_folder = Path("tmp_slices")
            if slices_folder.exists():
                shutil.rmtree(slices_folder)
            slices_folder.mkdir(parents=True, exist_ok=True)

            leaf_scan = LeafScan(
                view_config=view_config,
                leaf_config=separator_config,
                output_folder=slices_folder,
                display=False
            )

            # Adjust output paths (_sm_ → _so_)
            video_output_path = out_folder / video_path.name.replace("_sm_", "_so_")
            stacked_slices_path = video_output_path.with_suffix(".jpg")

            pred_remaining_area = leaf_scan.scanVideo(
                remaining_leaf_length=rough_remaining_length,
                video_path=str(video_path),
                output_path=str(video_output_path)
            )
            stitch_images_vertically(slices_folder, stacked_slices_path)

            # Calculate defoliation (%)
            pred_defoliation = (1 - pred_remaining_area / pred_original_area) * 100

            row = {
                "Field_ID": f_id,
                "Plant_ID": p_id,
                "Leaf_ID": l_id,
                "Media_ID": m_id,
                "Is_Defoliated": 1,

                "pred_original_area": pred_original_area,
                "true_original_area": true_original_area,

                "pred_remaining_area": pred_remaining_area,
                "true_remaining_area": true_remaining_area,

                "pred_defoliation": pred_defoliation,
                "true_defoliation": true_defoliation,
            }
            append_result(row)
            save_progress(i + 1)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            save_progress(i + 1)
            continue

    print("✅ Batch processing finished.")

if __name__ == "__main__":
    main()

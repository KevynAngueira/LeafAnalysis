# Author: Kevyn Angueira Irizarry
# Created: 2025-09-24
# Last Modified: 2025-09-24

import json
import joblib
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import LeafScan
from LeafScan.Configs import ViewExtractorConfig, LeafExtractorConfig
from LeafScan.Utils import stitch_images_vertically
from LeafScan.Utils.GetLeafModelData import GetLeafModelData

# === Config ===
MODEL_PATH = "SavedModels/gradient_boosting_model.pkl"
RESULTS_FILE = "batch_defoliation_results.csv"
PROGRESS_FILE = "progress.txt"

BASE_DIR = Path("/home/icicle/Research Datasets/LeafScan-CornDefoliation2025")

# === Utilities ===
def noisy_round(value, precision):
    noise = np.random.uniform(-precision, precision, size=np.shape(value))
    return np.round((value + noise) / precision) * precision

def get_all_defoliated_videos():
    """Traverse dataset for all defoliated videos."""
    return sorted(BASE_DIR.glob("field_*/plant_*/leaf_*/defoliated/media/*.mp4"))

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

    start_index = load_progress()
    print(f"▶️ Resuming at index {start_index}")

    for i, video_path in enumerate(videos[start_index:], start=start_index):
        try:
            print(f"\n📹 [{i+1}/{len(videos)}] Processing {video_path}")

            media_folder = video_path.parent  # .../media/
            status_folder = media_folder.parent  # .../defoliated/
            out_folder = status_folder / "output"
            out_folder.mkdir(parents=True, exist_ok=True)

            # True remaining length
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

            video_output_path = out_folder / video_path.name
            stacked_slices_path = video_output_path.with_suffix(".jpg")

            calculated_remaining_area = leaf_scan.scanVideo(
                remaining_leaf_length=rough_remaining_length,
                video_path=str(video_path),
                output_path=str(video_output_path)
            )
            stitch_images_vertically(slices_folder, stacked_slices_path)

            # Predict original area using widths + length (from Excel dataset features)
            # NOTE: you’ll need a reliable mapping between this video’s leaf and the Excel row
            # Here I just grab the "mean row" as placeholder
            # Replace with actual ID-based lookup if available
            widths = X_full.iloc[0].drop("Length").values  # placeholder widths
            length = rough_remaining_length
            X_pred = pd.DataFrame([list(widths) + [length]], columns=X_full.columns)
            estimated_original_area = model.predict(X_pred)[0]

            estimated_defoliation = (1 - calculated_remaining_area / estimated_original_area) * 100

            row = {
                "video": str(video_path),
                "estimated_original_area": estimated_original_area,
                "calculated_remaining_area": calculated_remaining_area,
                "estimated_defoliation": estimated_defoliation,
                "true_remaining_length": true_remaining_length
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

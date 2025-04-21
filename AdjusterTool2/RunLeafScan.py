# Author: Kevyn Angueira Irizarry
# Created: 2025-03-26
# Last Modified: 2025-04-21

import random
import shutil
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from Scripts.LeafScan import LeafScan
from DefoliationModeller.LeafData import LeafData

BASE_DIR = Path("LeafMedia")

def get_available_leaf_ids():
    return sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()])

def get_status_folder(leaf_id):
    return [p.name for p in (BASE_DIR / leaf_id).iterdir() if p.is_dir() and p.name in ["healthy", "defoliated"]]

def list_video_files(leaf_id, status):
    return sorted((BASE_DIR / leaf_id / status / "videos").glob("*.mp4"))

def prompt_selection(prompt_msg, options):
    print(prompt_msg)
    for i, opt in enumerate(options):
        print(f"{i+1}: {opt}")
    while True:
        try:
            choice = int(input("Select a number: ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
        except ValueError:
            pass
        print("Invalid choice. Try again.")

def parse_filename(filename):
    """
    Expects: leaf-<leaf_id>_<status>-<def_pct>_<media_type>-<media_id>.ext
    Example: leaf-002_defoliated-80_vid-01.mp4
    """
    name = filename.stem
    parts = name.split("_")
    leaf_id = parts[0].split("-")[1]
    status, def_pct = parts[1].split("-")
    media_type, media_id = parts[2].split("-")
    return leaf_id, status, int(def_pct), media_type, media_id

def clear_folder(folder):
    if folder.exists():
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

def main():
    # Step 1: Prompt for Leaf ID
    leaf_ids = get_available_leaf_ids()
    if not leaf_ids:
        print("❌ No leaf folders found in LeafMedia.")
        return
    leaf_id = prompt_selection("📂 Select a leaf ID:", leaf_ids)

    # Step 2: Prompt for Status (healthy / defoliated)
    statuses = get_status_folder(leaf_id)
    if not statuses:
        print("❌ No 'healthy' or 'defoliated' folders under this leaf.")
        return
    status = prompt_selection("🌿 Select the status:", statuses)

    # Step 3: Prompt for video
    video_files = list_video_files(leaf_id, status)
    if not video_files:
        print("❌ No videos found in this leaf/status folder.")
        return
    video_path = prompt_selection("🎥 Select a video:", video_files)

    # Step 4: Parse filename
    parsed_leaf_id, status_str, def_pct, _, media_id = parse_filename(video_path)
    parsed_leaf_id_int = int(parsed_leaf_id)

    results_root = BASE_DIR / leaf_id / status / "results" / media_id
    segment_folder = results_root / "leafSegments"
    output_folder = results_root / "output"

    # Step 5: Create folders if needed and clear if already exist
    for folder in [segment_folder, output_folder]:
        folder.mkdir(parents=True, exist_ok=True)
        clear_folder(folder)

    print(f"▶️ Running LeafScan on: {video_path}")
    print(f"📁 Segment output to: {segment_folder}")
    print(f"📁 Analysis output to: {output_folder}")

    # Step 6: Run LeafScan and get calculated area
    leafScan = LeafScan(output_folder=segment_folder, display=True)
    calculated_remaining_area, calculated_base_widths = leafScan.scanVideo(str(video_path), f"{str(output_folder)}/test.mp4")
   
    # Step 7: Get original area using LeafData
    leafData = LeafData()
    original_area_df = leafData.getAreaByID(parsed_leaf_id_int)
    original_area = float(original_area_df)

    # Step 8: Compute expected remaining area
    if status == "healthy":
        real_remaining_area = original_area
    else:
        real_remaining_area = original_area * (1 - def_pct / 100)

    # Step 9: Percent change between calculated and expected remaining area
    if real_remaining_area > 0:
        remaining_area_pchange = (calculated_remaining_area - real_remaining_area) / real_remaining_area * 100
    else:
        remaining_area_pchange = 0.0

    # Step 10: Compute the base widths and length esimation (noise)
    #widths = leafData.getLeafByID(parsed_leaf_id_int)["Start_Width"].tolist()[1:4]
    widths = calculated_base_widths
    length = leafData.getLengthByID(parsed_leaf_id_int)

    np.random.seed(42)
    noise = np.random.uniform(-0.25, 0.25)
    rough_length = round((length + noise) * 2) / 2

    # Step 11: Compute estimate of the original leaf area
    model_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/AreaEstimation/SavedModels/gradient_boosting_model.pkl"
    gb_model = joblib.load(model_path)

    X_pred = pd.DataFrame([widths + [rough_length]], columns=["width_0", "width_1", "width_2", "length"])
    estimated_original_area = gb_model.predict(X_pred)[0]

    # Step 12: Percent change between estimated and expected original area
    original_area_pchange = ((estimated_original_area - original_area) / original_area) * 100

    # Step 13: Compute estimated defoliation percentage
    estimated_defoliation = (1 - calculated_remaining_area / estimated_original_area) * 100

    # Step 14: Percent change between estimated and expected defoliation percentage
    real_defoliation = def_pct
    defoliation_pchange = ((estimated_defoliation - real_defoliation) / real_defoliation) * 100

    # === Report ===
    print("\n📊 --- Estimation Summary ---")
    print(f"\n🌱 True Original Area: {original_area:.2f}")
    print(f"📏 Estimated Original Area: {estimated_original_area:.2f} ({original_area_pchange:+.2f}% change)")

    print(f"\n🌱 True Remaining Area: {real_remaining_area:.2f}")
    print(f"📏 Scanned Remaining Area: {calculated_remaining_area:.2f} ({remaining_area_pchange:+.2f}% change)")

    print(f"\n🌱 True Defoliation: {real_defoliation:.2f}%")
    print(f"📏 Estimated Defoliation: {estimated_defoliation:.2f}% ({defoliation_pchange:+.2f}% change)")


if __name__ == "__main__":
    main()

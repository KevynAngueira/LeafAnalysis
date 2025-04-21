# Author: Kevyn Angueira Irizarry
# Created: 2025-04-21
# Last Modified: 2025-04-21

# Author: Kevyn Angueira Irizarry
# Batch evaluation with MAE + resume + outlier save + error tolerance + skip leaf IDs < 6

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

from Scripts.LeafScan import LeafScan
from DefoliationModeller.LeafData import LeafData

BASE_DIR = Path("LeafMedia")
MODEL_PATH = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/AreaEstimation/SavedModels/gradient_boosting_model.pkl"
RESULTS_FILE = "batch_defoliation_results.csv"
OUTLIERS_FILE = "batch_outliers.txt"
PROGRESS_FILE = "progress.txt"

def parse_filename(filename):
    name = filename.stem
    parts = name.split("_")
    leaf_id = parts[0].split("-")[1]
    status, def_pct = parts[1].split("-")
    media_type, media_id = parts[2].split("-")
    return leaf_id, status, int(def_pct), media_type, media_id

def get_all_defoliated_videos():
    video_paths = []
    for leaf_dir in sorted(BASE_DIR.iterdir()):
        if not leaf_dir.is_dir():
            continue
        leaf_id_str = leaf_dir.name
        if not leaf_id_str.isdigit() or int(leaf_id_str) < 6:
            continue  # 🚫 Ignore leaf IDs less than 6
        defoliated_dir = leaf_dir / "defoliated" / "videos"
        if not defoliated_dir.exists():
            continue
        video_paths.extend(sorted(defoliated_dir.glob("*.mp4")))
    return video_paths

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
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_progress(index):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))

def save_outlier(video_path):
    with open(OUTLIERS_FILE, "a") as f:
        f.write(f"{video_path}")

def append_result(row):
    header = not Path(RESULTS_FILE).exists()
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode='a', index=False, header=header)

def main():
    leafScan = LeafScan()
    leafData = LeafData()
    gb_model = joblib.load(MODEL_PATH)

    videos = get_all_defoliated_videos()
    if not videos:
        print("❌ No defoliated videos found.")
        return

    start_index = 0
    if Path(PROGRESS_FILE).exists():
        choice = prompt_resume_options()
        if choice == 1:
            start_index = load_progress()
        elif choice == 2:
            start_index = 0
            Path(RESULTS_FILE).unlink(missing_ok=True)
            Path(OUTLIERS_FILE).unlink(missing_ok=True)
        elif choice == 3:
            start_index = int(input("Enter custom start index (0-based): "))

    total_videos = len(videos)
    processed = 0
    outliers = []

    for i, video_path in enumerate(videos[start_index:], start=start_index):
        print(f"📹 [{i+1}/{total_videos}] Processing {video_path.name}")
        try:
            leaf_id_str, status, def_pct, _, media_id = parse_filename(video_path)
            leaf_id = int(leaf_id_str)

            if leaf_id < 6:
                print(f"⚠️ Skipping leaf ID {leaf_id} (below threshold)")
                save_progress(i + 1)
                continue

            original_area = float(leafData.getAreaByID(leaf_id))
            if original_area == 0:
                raise ValueError("Original area is 0")

            real_remaining_area = original_area * (1 - def_pct / 100)

            segment_folder = BASE_DIR / leaf_id_str / "defoliated" / "results" / media_id / "leafSegments"
            output_folder = BASE_DIR / leaf_id_str / "defoliated" / "results" / media_id / "output"
            segment_folder.mkdir(parents=True, exist_ok=True)
            output_folder.mkdir(parents=True, exist_ok=True)

            calculated_remaining_area, calculated_base_widths = leafScan.scanVideo(str(video_path), f"{str(output_folder)}/test.mp4")

            widths = calculated_base_widths
            length = leafData.getLengthByID(leaf_id)
            np.random.seed(42)
            noise = np.random.uniform(-0.25, 0.25)
            rough_length = round((length + noise) * 2) / 2

            X_pred = pd.DataFrame([widths + [rough_length]], columns=["width_0", "width_1", "width_2", "length"])
            estimated_original_area = gb_model.predict(X_pred)[0]

            estimated_defoliation = (1 - calculated_remaining_area / estimated_original_area) * 100
            real_defoliation = def_pct

            # Sanity check: prevent extreme miscalculations
            if abs(calculated_remaining_area - real_remaining_area) > 2 * original_area:
                print("⚠️ Error: Remaining area scan anomaly")
                save_outlier(video_path)
                outliers.append(video_path)

            row = {
                "leaf_id": leaf_id,
                "media_id": media_id,
                "real_original_area": original_area,
                "estimated_original_area": estimated_original_area,
                "real_remaining_area": real_remaining_area,
                "calculated_remaining_area": calculated_remaining_area,
                "real_defoliation": real_defoliation,
                "estimated_defoliation": estimated_defoliation
            }

            append_result(row)
            save_progress(i + 1)
            processed += 1

        except Exception as e:
            print(f"⚠️ Error: {e}")
            save_progress(i + 1)
            continue

    if not Path(RESULTS_FILE).exists():
        print("⚠️ No valid results were saved.")
        return

    df = pd.read_csv(RESULTS_FILE)
    print("📊 --- Final Summary ---")

    def print_mae(title, data):
        print(f"{title}")
        print(f"🟩 MAE - Original Area: {np.mean(np.abs(data['estimated_original_area'] - data['real_original_area'])):.2f}")
        print(f"🟨 MAE - Remaining Area: {np.mean(np.abs(data['calculated_remaining_area'] - data['real_remaining_area'])):.2f}")
        print(f"🟥 MAE - Defoliation %: {np.mean(np.abs(data['estimated_defoliation'] - data['real_defoliation'])):.2f}")

    print_mae("📁 ALL DATA", df)

    outliers_path = Path(OUTLIERS_FILE)
    if outliers_path.exists():
        with open(outliers_path) as f:
            # Extract just the filenames from the full paths
            outlier_filenames = set(Path(line.strip()).name for line in f if line.strip())

        df_filtered = df[~df.apply(
            lambda row: f"leaf-{int(row['leaf_id']):03d}_defoliated-{int(row['real_defoliation'])}_vid-{int(row['media_id']):02d}.mp4" in outlier_filenames,
            axis=1
        )]

        print_mae("📁 WITHOUT OUTLIERS", df_filtered)
        print(f"✅ Processed {processed} new videos.")
        print(f"🚫 Outliers recorded: {len(outlier_filenames)} → loaded from {OUTLIERS_FILE}")
    else:
        print("🚫 No outlier file found.")

if __name__ == "__main__":
    main()

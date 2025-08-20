# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import os
import shutil
from pathlib import Path
from Scripts.LeafScan import LeafScan

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
    return leaf_id, status, def_pct, media_type, media_id

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

    # Step 4: Parse filename and prepare paths
    _, _, _, _, media_id = parse_filename(video_path)

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

    leafScan = LeafScan(output_folder=segment_folder)
    leafScan.scanVideo(str(video_path), f"{str(output_folder)}/test.mp4")

if __name__ == "__main__":
    main()

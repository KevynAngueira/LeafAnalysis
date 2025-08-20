# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

# Author: Kevyn Angueira Irizarry
# Purpose: Step-by-step file selector for new dataset structure

from pathlib import Path

# Set your dataset root here
DATASET_ROOT = Path("/home/icicle/Research Datasets/LeafScan-CornDefoliation2025/Private/LeafScan-CornDefoliation2025-v1/data")

def prompt_selection(prompt_msg, options):
    print(f"
{prompt_msg}")
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

def get_field_ids():
    return sorted([p.name for p in DATASET_ROOT.glob("field_*") if p.is_dir()])

def get_plant_ids(field_path):
    return sorted([p.name for p in field_path.glob("plant_*") if p.is_dir()])

def get_leaf_ids(plant_path):
    return sorted([p.name for p in plant_path.glob("leaf_*") if p.is_dir()])

def get_statuses(leaf_path):
    return sorted([p.name for p in leaf_path.iterdir() if p.is_dir() and p.name in ["healthy", "defoliated", "simulated"]])

def get_video_files(status_path):
    media_folder = status_path / "media"
    return sorted(media_folder.glob("*.mp4"))

# Helper: maps ID number (int) → full folder name
def build_id_map(entries, prefix):
    id_map = {}
    for entry in entries:
        try:
            id_num = int(entry.replace(f"{prefix}_", ""))
            id_map[id_num] = entry
        except ValueError:
            continue
    return id_map

# New prompt function that uses numeric keys
def prompt_id_selection(prompt_msg, id_map):
    print(f"
{prompt_msg}")
    for id_num, name in sorted(id_map.items()):
        print(f"{id_num}: {name}")
    while True:
        try:
            choice = int(input("Type the numeric ID: "))
            if choice in id_map:
                return id_map[choice]
        except ValueError:
            pass
        print("Invalid ID. Try again.")

def main():
    # Step 1: Select Field by ID
    field_names = get_field_ids()
    field_map = build_id_map(field_names, "field")
    if not field_map:
        print("❌ No field folders found.")
        return
    selected_field = prompt_id_selection("🌾 Select a field by ID:", field_map)
    field_path = DATASET_ROOT / selected_field

    # Step 2: Select Plant by ID
    plant_names = get_plant_ids(field_path)
    plant_map = build_id_map(plant_names, "plant")
    if not plant_map:
        print("❌ No plant folders found in this field.")
        return
    selected_plant = prompt_id_selection("🪴 Select a plant by ID:", plant_map)
    plant_path = field_path / selected_plant

    # Step 3: Select Leaf by ID
    leaf_names = get_leaf_ids(plant_path)
    leaf_map = build_id_map(leaf_names, "leaf")
    if not leaf_map:
        print("❌ No leaf folders found in this plant.")
        return
    selected_leaf = prompt_id_selection("🍃 Select a leaf by ID:", leaf_map)
    leaf_path = plant_path / selected_leaf

    # Step 4–5 remain unchanged
    statuses = get_statuses(leaf_path)
    if not statuses:
        print("❌ No valid status folders.")
        return
    selected_status = prompt_selection("🌿 Select a leaf status:", statuses)
    status_path = leaf_path / selected_status

    video_files = get_video_files(status_path)
    if not video_files:
        print("❌ No video files found in media/ folder.")
        return
    selected_video = prompt_selection("🎥 Select a video file:", [f.name for f in video_files])
    selected_video_path = status_path / "media" / selected_video

    print(f"
✅ Selected video: {selected_video_path}")
    return selected_video_path

if __name__ == "__main__":
    main()

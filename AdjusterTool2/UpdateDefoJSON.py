# Author: Kevyn Angueira Irizarry
# Created: 2025-04-21
# Last Modified: 2025-04-21

import csv
import json
from pathlib import Path

BASE_DIR = Path("LeafMedia")

def load_csv_data(csv_path):
    data = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                leaf_id = row["leaf_id"].zfill(3)
                plant_id = row["plant_id"]
                slice_pct = float(row["slice_percentage"])
                cut_pct = float(row["cut_percentage"])
                data[leaf_id] = {
                    "plant_id": plant_id,
                    "slice_percentage": slice_pct,
                    "cut_percentage": cut_pct
                }
            except Exception as e:
                print(f"⚠️ Skipping invalid row {row}: {e}")
    return data

def update_json_file(json_path, slice_pct, cut_pct, plant_id):
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        metadata["slice_percentage"] = slice_pct
        metadata["cut_percentage"] = cut_pct
        metadata["plant_id"] = plant_id

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Updated {json_path}")
    except Exception as e:
        print(f"❌ Failed to update {json_path}: {e}")

def update_leaf_metadata(leaf_id, slice_pct, cut_pct, plant_id):
    leaf_path = BASE_DIR / leaf_id
    if not leaf_path.exists():
        print(f"⚠️ Leaf ID {leaf_id} not found in {BASE_DIR}")
        return

    for status in ["healthy", "defoliated"]:
        for media_type in ["images", "videos"]:
            json_dir = leaf_path / status / media_type
            if not json_dir.exists():
                continue
            for json_file in json_dir.glob("*.json"):
                if status == "healthy":
                    update_json_file(json_file, 0, 0, plant_id)
                else:
                    update_json_file(json_file, slice_pct, cut_pct, plant_id)

def main():
    csv_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/leaf_data.csv"
    if not Path(csv_path).is_file():
        print("❌ Invalid CSV file path.")
        return

    data = load_csv_data(csv_path)
    for leaf_id, info in data.items():
        update_leaf_metadata(
            leaf_id,
            info["slice_percentage"],
            info["cut_percentage"],
            info["plant_id"]
        )

if __name__ == "__main__":
    main()

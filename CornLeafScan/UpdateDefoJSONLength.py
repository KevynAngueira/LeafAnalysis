# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20

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
                healthy_length = float(row["healthy_length"])
                defoliated_length = float(row["defoliated_length"])
                data[leaf_id] = {
                    "healthy_length": healthy_length,
                    "defoliated_length": defoliated_length
                }
            except Exception as e:
                print(f"⚠️ Skipping invalid row {row}: {e}")
    return data

def update_json_file(json_path, original_length, remaining_length):
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Update the correct field
        metadata["original_length"] = original_length
        metadata["remaining_length"] = remaining_length

        for key in ['healthy_length', 'defoliated_length']: 
            metadata.pop(key, None)

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Updated {json_path} (original_length = {original_length} AND remaining_length = {remaining_length})")
    except Exception as e:
        print(f"❌ Failed to update {json_path}: {e}")

def update_leaf_metadata(leaf_id, healthy_length, defoliated_length):
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
                    update_json_file(json_file, healthy_length, healthy_length)
                else:
                    update_json_file(json_file, healthy_length, defoliated_length)

def main():
    csv_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/leaf_data_lengths.csv"
    if not Path(csv_path).is_file():
        print("❌ Invalid CSV file path.")
        return

    data = load_csv_data(csv_path)
    for leaf_id, info in data.items():
        update_leaf_metadata(
            leaf_id,
            info["healthy_length"],
            info["defoliated_length"]
        )

if __name__ == "__main__":
    main()

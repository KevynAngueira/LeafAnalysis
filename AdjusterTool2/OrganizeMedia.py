# Author: Kevyn Angueira Irizarry
# Created: 2025-03-26
# Last Modified: 2025-03-26

import os
import shutil
from datetime import datetime
import json
from pathlib import Path
import mimetypes

BASE_DIR = Path("LeafMedia")
NEW_MEDIA_DIR = Path("NewMedia")  # The folder where new media is temporarily stored

def get_next_leaf_id():
    existing_ids = [
        int(folder.name)
        for folder in BASE_DIR.glob("*")
        if folder.is_dir() and folder.name.isdigit()
    ]
    return str(max(existing_ids) + 1).zfill(3) if existing_ids else "000"

def get_next_media_id(folder, media_type):
    existing = [
        f.name for f in folder.glob(f"*_{media_type}-*.json")
    ]
    return str(len(existing)).zfill(2)

def list_files_in_directory(directory):
    files = [f for f in directory.iterdir() if f.is_file()]
    return files

def select_media_file():
    media_files = list_files_in_directory(NEW_MEDIA_DIR)

    if not media_files:
        print("No media files found in the NewMedia folder.")
        return None

    print("Select a media file from the NewMedia folder:")
    for i, file in enumerate(media_files, 1):
        print(f"{i}. {file.name}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(media_files):
                return media_files[choice - 1]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def sanitize_inputs():
    # Get the media file from the user selection
    media_path = select_media_file()
    if not media_path:
        return None, None, None

    # Prompt for defoliation status (0 = healthy, 1 = defoliated)
    defoliation_status_input = input("Enter defoliation status (0 = healthy, 1 = defoliated): ").strip()
    while defoliation_status_input not in ['0', '1']:
        defoliation_status_input = input("Invalid input. Enter 0 for healthy or 1 for defoliated: ").strip()

    defoliation_status = int(defoliation_status_input)

    # Set defoliation percentage
    if defoliation_status == 0:
        def_percent = 0  # Healthy, no defoliation
    else:
        def_percent_input = input("Enter defoliation percentage (0-100): ").strip()
        while not def_percent_input.isdigit() or not (0 <= int(def_percent_input) <= 100):
            def_percent_input = input("Invalid input. Enter defoliation percentage as a number between 1 and 100: ").strip()
        def_percent = int(def_percent_input)

    # Get or auto-generate the leaf ID
    leaf_id = input("Enter leaf ID (leave blank for auto): ").strip()
    if not leaf_id:
        leaf_id = get_next_leaf_id()
    else:
        leaf_id = leaf_id.zfill(3)

    return media_path, leaf_id, defoliation_status, def_percent

def organize_and_move(media_path, leaf_id, def_status, def_percent):
    file_ext = Path(media_path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(media_path)

    if mime_type is None:
        print("Could not determine file type.")
        return

    is_video = mime_type.startswith("video")
    media_type = "vid" if is_video else "img"
    file_type_folder = "videos" if is_video else "images"
    status = "healthy" if def_status == 0 else "defoliated"
    
    # Create full directory path
    full_path = BASE_DIR / leaf_id / status / file_type_folder
    full_path.mkdir(parents=True, exist_ok=True)

    # Determine next media id
    media_id = get_next_media_id(full_path, media_type)

    # Build filename
    filename_base = f"leaf-{leaf_id}_{status}-{def_percent}_{media_type}-{media_id}"
    media_filename = f"{filename_base}{file_ext}"
    json_filename = f"{filename_base}.json"

    dest_media_path = full_path / media_filename
    dest_json_path = full_path / json_filename

    # Move the media file
    shutil.move(media_path, dest_media_path)

    # Create JSON metadata
    metadata = {
        "leaf_id": leaf_id,
        "status": status,
        "defoliation_percent": def_percent,
        "media_type": media_type,
        "media_id": media_id,
        "creation_date": datetime.now().strftime("%Y-%m-%d")
    }

    with open(dest_json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ File moved to: {dest_media_path}")
    print(f"📄 Metadata saved to: {dest_json_path}")

def main():
    media_path, leaf_id, def_status, def_percent = sanitize_inputs()
    if media_path:
        organize_and_move(media_path, leaf_id, def_status, def_percent)
    else:
        print("No valid media file selected. Exiting.")

if __name__ == "__main__":
    main()

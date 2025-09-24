# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import re
import cv2
import numpy as np
from pathlib import Path

def stitch_images_vertically(input_folder: Path, output_path: Path):
    # Step 1: Get all image paths sorted alphanumerically
    #image_paths = sorted(input_folder.glob("*.*"))  # includes .jpg, .png, etc.
    #image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    def extract_number(path):
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else -1

    image_paths = sorted(
        [p for p in input_folder.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=extract_number
    )

    print(image_paths)

    if not image_paths:
        print("❌ No images found in the folder.")
        return

    # Step 2: Load all images
    images = [cv2.imread(str(p)) for p in image_paths]
    
    # Step 3: Optionally resize to same width (assumes first image width)
    base_width = images[0].shape[1]
    resized_images = [
        cv2.resize(img, (base_width, int(img.shape[0] * base_width / img.shape[1])))
        if img.shape[1] != base_width else img
        for img in images
    ]

    # Step 4: Stack vertically
    stitched = np.vstack(resized_images)

    # Step 5: Save result
    cv2.imwrite(str(output_path), stitched)
    print(f"✅ Saved stitched image to: {output_path}")

if __name__ == "__main__":
    input_folder = Path("tmp_slices")        # Change as needed
    output_path = Path("tmp_out/output.jpg")      # Output file
    stitch_images_vertically(input_folder, output_path)

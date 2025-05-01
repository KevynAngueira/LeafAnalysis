# Author: Kevyn Angueira Irizarry
# Created: 2025-03-04
# Last Modified: 2025-05-01

import os
import cv2
import numpy as np
import glob
import re

import pandas as pd

#from Scripts.ResizeForDisplay import resize_for_display

def resize_for_display(image, height=600):
    # Rotate image if width is greater than height
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]

    # Resize while maintaining aspect ratio
    scale = height / h
    return cv2.resize(image, (int(w * scale), height))

def stitch_images(input_folder, output_folder, output_filename="stitched_leaf_segments.png", reverse_order=False):
    target_dimensions = (1, 6.5)

    # Get all images matching the pattern
    input_path = os.path.join(input_folder, "frame_*.jpg")
    image_files = glob.glob(input_path)
    
    # Extract numbers and sort by numerical order
    image_files.sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))

    # Optionally reverse the order
    if reverse_order:
        image_files = list(reversed(image_files))
    
    if not image_files:
        print("No images found!")
        return None

    # Load images
    images = [cv2.imread(img) for img in image_files]
    
    # Stitch vertically
    stitched_image = np.vstack(images)
    
    # Save result
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, stitched_image)
    
    leaf_image = cv2.imread(output_path)
    #cv2.imshow("Reconstructed Leaf", leaf_image)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return leaf_image

if __name__ == "__main__":
    # Load the batch defoliation results
    results_path = "/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/batch_defoliation_results.csv"
    results_df = pd.read_csv(results_path)


    for i in range(6,27):
        leaf_id = f'{i:03}'
        vid_id = "00"

        image_folder = f"/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/LeafMedia/{leaf_id}/defoliated/images/"
        input_folder = f"/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/LeafMedia/{leaf_id}/defoliated/results/{vid_id}/leafSegments"
        output_folder = f"/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/LeafMedia/{leaf_id}/defoliated/results/{vid_id}/output"

        files = glob.glob(f"{image_folder}/*jpg")
        original_image_file = files[0] if files else None

        if original_image_file is None:
            print(f"Error: No original image. Skipping {leaf_id}")
            continue

        original_image = cv2.imread(original_image_file)

        regular_image = stitch_images(input_folder, output_folder)
        reversed_image = stitch_images(input_folder, output_folder, reverse_order=True)
        
        if regular_image is None: 
            print(f"Error: No segments to reconstruct. Skipping {leaf_id}")
            continue

        # Resize all images to same height
        original_resized = resize_for_display(original_image)
        regular_resized = resize_for_display(regular_image)
        reversed_resized = resize_for_display(reversed_image)

        # Concatenate images side by side
        combined = cv2.hconcat([original_resized, regular_resized, reversed_resized])

        # Extract numeric leaf ID for lookup
        numeric_leaf_id = int(leaf_id)
        row = results_df[(results_df["leaf_id"] == numeric_leaf_id) & (results_df["media_id"] == int(vid_id))]

        if not row.empty:
            real_o = row["real_original_area"].values[0]
            est_o = row["estimated_original_area"].values[0]
            real_r = row["real_remaining_area"].values[0]
            calc_r = row["calculated_remaining_area"].values[0]
            real_d = row["real_defoliation"].values[0]
            est_d = row["estimated_defoliation"].values[0]

            print(f"\nLeaf {leaf_id} Video {vid_id}")
            print(f"  Original Area Error: {est_o - real_o:.4f} ({100 * (est_o - real_o)/real_o:.2f}%)")
            print(f"  Remaining Area Error: {calc_r - real_r:.4f} ({100 * (calc_r - real_r)/real_r:.2f}%)")
            print(f"  Defoliation Error: {est_d - real_d:.2f} ({100 * (est_d - real_d)/real_d:.2f}%)")
        else:
            print(f"[{leaf_id}] No matching data in CSV for media_id {vid_id}")

        print("Displaying image window...")
        cv2.imshow(f"Leaf {leaf_id} - Original | Reconstructed | Reversed", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
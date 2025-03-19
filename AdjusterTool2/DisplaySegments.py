# Author: Kevyn Angueira Irizarry
# Created: 2025-03-19
# Last Modified: 2025-03-19

import os
import cv2
import numpy as np
import glob
import re

def stitch_images(input_folder, output_folder, output_filename="stitched_leaf_segments.png"):
    target_dimensions = (1, 6.5)

    # Get all images matching the pattern
    input_path = os.path.join(input_folder, "frame_*.jpg")
    image_files = glob.glob(input_path)
    
    # Extract numbers and sort by numerical order
    image_files.sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
    
    if not image_files:
        print("No images found!")
        return

    # Load images
    images = [cv2.imread(img) for img in image_files]
    
    # Stitch vertically
    stitched_image = np.vstack(images)
    
    # Save result
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, stitched_image)
    
    leaf_image = cv2.imread(output_path)
    cv2.imshow("Reconstructed Leaf", leaf_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_folder = "demonstration0"
    input_folder = f"/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/LeafSegments"
    output_folder = f"/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos"
    stitch_images(input_folder, output_folder)

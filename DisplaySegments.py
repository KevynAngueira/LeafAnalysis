import os
import cv2
import numpy as np
import glob
import re

from LeafArea import calculateArea

def stitch_images(input_folder, output_folder, output_filename="stitched_leaf_segments.png"):
    target_dimensions = (1, 6.5)

    # Get all images matching the pattern
    input_path = os.path.join(input_folder, "leaf_segment_*.png")
    image_files = glob.glob(input_path)
    
    # Extract numbers and sort by numerical order
    image_files.sort(key=lambda x: int(re.search(r'leaf_segment_(\d+)', x).group(1)))
    
    if not image_files:
        print("No images found!")
        return
    
    total_area = 0
    for image in image_files:
        image_area = calculateArea(image, target_dimensions)
        print(f"{image} Area: {image_area}")   
        total_area += image_area     

    # Load images
    images = [cv2.imread(img) for img in image_files]
    
    # Stitch vertically
    stitched_image = np.vstack(images)
    
    # Save result
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, stitched_image)

    print(f"✅ Stitched image saved as {output_path}")
    print(f"Total Area: {total_area}")
    
    leaf_image = cv2.imread(output_path)
    cv2.imshow("Reconstructed Leaf", leaf_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return total_area

if __name__ == "__main__":
    video_folder = "demonstration0"
    input_folder = f"LeafSegments/{video_folder}"
    output_folder = f"TestVideos/{video_folder}"
    stitch_images(input_folder, output_folder)

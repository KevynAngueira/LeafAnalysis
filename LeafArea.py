import cv2
import numpy as np

def calculateArea(image_path, target_dimensions):
    image = cv2.imread(image_path)

    h, w, _ = image.shape
    target_h, target_w = target_dimensions

    total_image_pixels =  h * w

    non_black_mask = np.any(image > 0, axis=-1)
    non_black_pixels = np.count_nonzero(non_black_mask)
    
    non_black_ratio = non_black_pixels / total_image_pixels
    print(non_black_ratio)

    target_area = (h/100) * target_w
    print(target_area)

    non_black_area = non_black_ratio * target_area

    return non_black_area
    

image_path = "LeafSegments/stitched_leaf_CCORR.png"
target_dimensions = (1, 6.5)

result_area = calculateArea(image_path, target_dimensions)
print(result_area)
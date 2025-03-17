import cv2
import numpy as np
from DetectRectangle import imagePreprocessing, applyMask
from Helper.ResizeForDisplay import resize_for_display

def extractGreen(target_image, rectangle_hsv_thresholds, leaf_hsv_thresholds, rectangle_dimensions, display=False):

    rect_lower_thres, rect_upper_thresh = rectangle_hsv_thresholds
    leaf_lower_thres, leaf_upper_thresh = leaf_hsv_thresholds

    preproc_image = imagePreprocessing(target_image, blur=(3,3))
    
    rect_mask = applyMask(preproc_image, rect_lower_thres, rect_upper_thresh)
    leaf_mask = applyMask(preproc_image, leaf_lower_thres, leaf_upper_thresh, False, morph_iterations=2, kernel_size=(3,3))

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only the largest contour (assuming it's the leaf)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        refined_leaf_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(refined_leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        refined_leaf_mask = leaf_mask  # If no contours, fallback to color mask
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(target_image, target_image, mask=refined_leaf_mask)

    # Derive the background mask
    exclusion_mask = cv2.bitwise_or(rect_mask, refined_leaf_mask)
    background_mask = cv2.bitwise_not(exclusion_mask)

    # Count pixels in each mask
    total_leaf_pixels = cv2.countNonZero(refined_leaf_mask)
    total_background_pixels = cv2.countNonZero(background_mask)

    #print(f"Total Leaf Pixels: {total_leaf_pixels}")
    #print(f"Total Background Pixels: {total_background_pixels}")

    # Calculate leaf area
    total_target_pixels = total_leaf_pixels + total_background_pixels
    w, h = rectangle_dimensions
    total_target_area = w*h

    if total_target_pixels > 0:
        target_pixel_ratio = total_leaf_pixels / total_target_pixels
        leaf_area = total_target_area * target_pixel_ratio
    else:
        target_pixel_ratio = 0
        leaf_area = 0

    #print(f"Total Pixels: {total_target_pixels}")
    #print(f"Total Area: {total_target_area}")
    #print(f"Pixel Ratio: {target_pixel_ratio}")
    print(f"Leaf Area: {leaf_area}")

    # Display results
    if display:
        #cv2.imshow("Rectangle Mask", resize_for_display(rect_mask))
        #cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
        cv2.imshow("Refined Leaf Mask", resize_for_display(refined_leaf_mask))
        cv2.imshow("Segmented Leaf", resize_for_display(result))
        cv2.imshow("Background Mask", resize_for_display(background_mask))

    return result, leaf_area
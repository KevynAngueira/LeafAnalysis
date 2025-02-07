import cv2
import numpy as np
from Helper.NormalizeBrightness import normalizeBrightness, dynamicValueThreshold
from Helper.ResizeForDisplay import resize_for_display


def imagePreprocessing(image, blur=(5,5)): 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize brightness
    hsv = normalizeBrightness(hsv)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(hsv, blur, 0)
    
    return blurred

def applyMask(hsv, lower_mask, upper_mask, dynamicVal=True, morph_iterations=2, kernel_size=(10,10)):
    # Dynamic Threshold?
    if dynamicVal:
        lower_v = dynamicValueThreshold(hsv)
        lower_mask[2] = lower_v # Tool lower value
        #print(lower_v)
    
    # Apply mask
    mask = cv2.inRange(hsv, lower_mask, upper_mask)

    # Morphological operations to remove small protrusions
    if morph_iterations > 0:
        kernel = np.ones(kernel_size, np.uint8)  # Adjust kernel size if necessary
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)  # Close small gaps
    
    return mask

def getRectangleContours(image, lower_mask, upper_mask, dynamicVal, draw=False):
    
    image = imagePreprocessing(image)
    mask = applyMask(image, lower_mask, upper_mask, dynamicVal)
    #cv2.imshow("Orange Mask", resize_for_display(mask))

    #mask_non_orange = cv2.bitwise_not(mask_orange)
    edges = cv2.Canny(mask, 50, 150)
    #cv2.imshow("Edges", resize_for_display(edges))

    # Find contours from Canny edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if draw: cv2.drawContours(image, contours, -1, (255, 255, 0), 2)

    return contours

def getTargetRectangle(contours, aspect_ratio_thresholds, draw=False, draw_target=None):
    # Track the largest contour that fits our target aspect ratio
    target_rect = None
    target_box = None
    max_area = 0

    for contour in contours:
        # Get bounding box for each contour
        min_rect = cv2.minAreaRect(contour)
        w, h = min_rect[1]  
        #print(f'({w}, {h})')

        box = cv2.boxPoints(min_rect)
        box = np.intp(box)
        
        aspect_ratio = max(w, h) / min(w, h) if min(w,h) != 0 else 0
        #print(aspect_ratio)

        # Check if box fits our expected aspect ratio
        lower_bound, upper_bound = aspect_ratio_thresholds
        if lower_bound <= aspect_ratio <= upper_bound:
            area = w*h
            if area > max_area:
                max_area = area
                target_box = box
                target_rect = min_rect
            if draw:
                cv2.drawContours(draw_target, [box], 0, (255, 0, 0), 2)
        else:
            if draw:
                cv2.drawContours(draw_target, [box], 0, (255, 0, 255), 2)
    
    return target_box, target_rect

def detectRectangle(image, color_thresholds, rectangle_dimensions, dynamicVal=True, draw=True):
    
    lower_mask, upper_mask = color_thresholds
    contours = getRectangleContours(image, lower_mask, upper_mask, dynamicVal)

    # Define aspect ratio thresholds
    w, h = rectangle_dimensions
    aspect_ratio = w/h if h != 0 else 0
    aspect_ratio_thresholds = (aspect_ratio-1.5, aspect_ratio+0.5)
    
    target_box, target_rect = getTargetRectangle(contours, aspect_ratio_thresholds, draw=draw, draw_target=image)

    return target_box, target_rect
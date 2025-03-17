import cv2
import numpy as np

from Helper.ResizeForDisplay import resize_for_display

def detect_background_type(image, threshold=60):
    """
    Determines whether the background is predominantly black/white or colored.
    Uses the mean saturation value of the image as a heuristic.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    return mean_saturation < threshold  # True if background is black/white

def apply_dynamic_hsv_mask(image):
    """
    Applies one of two HSV masks dynamically based on the background type.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if detect_background_type(image):
        # Background is mostly black/white, use saturation-based masking
        lower_bound = np.array([0, 60, 0])
        upper_bound = np.array([179, 255, 255])
    else:
        # Background has colors, use hue-based masking for pink
        lower_bound = np.array([140, 0, 0])
        upper_bound = np.array([179, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

if __name__ == "__main__":
    image_path = "/home/icicle/VSCode/LeafAnalysis/ColorTests/NewTool/NewTool2.jpg"  # Change this to your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image.")
    else:
        result, mask = apply_dynamic_hsv_mask(image)
        
        cv2.imshow("Original Image", resize_for_display(image))
        cv2.imshow("Mask", resize_for_display(mask))
        cv2.imshow("Extracted Object", resize_for_display(result))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np

def crop_leaf(frame, threshold=10):
    """
    Crop out the black borders around the leaf using thresholding and contour detection.
    Assumes that the leaf is the largest bright region in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, (0, 0, frame.shape[1], frame.shape[0])
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return frame[y:y+h, x:x+w], (x, y, w, h)

def pad_image_to_match(image, target_shape):
    """
    Pads an image with zeroes to match the target shape.
    """
    h, w = image.shape
    target_h, target_w = target_shape
    
    pad_h = max(20, h)
    pad_w = max(target_w, w)
    
    padded = np.zeros((pad_h, pad_w), dtype=np.uint8)
    start_y = (pad_h - h) // 2
    start_x = (pad_w - w) // 2
    
    padded[start_y:start_y + h, start_x:start_x + w] = image
    
    return padded, start_y

def extract_bottom_band(image, band_height=20):
    return image[-band_height:, :]

def stitch_images(image1, image2):
    """ 
    Stitch two images vertically using template matching but apply the results to the original images.
    """
    # Crop the leaf regions and get bounding boxes
    cropped1, bbox1 = crop_leaf(image1)
    cropped2, bbox2 = crop_leaf(image2)
    
    # Convert to grayscale for template matching
    gray1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)
    
    # Pad gray2 to match gray1's shape
    padded_gray2, pad_offset = pad_image_to_match(gray2, gray1.shape)
    
    # Extract the bottom band of gray1 as the template
    band_height = 20
    template_band = extract_bottom_band(gray1, band_height)
    
    # Template Matching
    result = cv2.matchTemplate(padded_gray2, template_band, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"Template matching score: {max_val:.2f}")
    overlap_height = max_loc[1] - pad_offset
    
    # Stitch using original images (not cropped)
    stitch_y = bbox1[1] + cropped1.shape[0]
    new_height = stitch_y + (image2.shape[0] - (bbox2[1] + overlap_height))
    stitched_width = max(image1.shape[1], image2.shape[1])
    
    stitched_image = np.zeros((new_height, stitched_width, 3), dtype=np.uint8)
    stitched_image[:image1.shape[0], :image1.shape[1]] = image1
    stitched_image[stitch_y:, :image2.shape[1]] = image2[bbox2[1] + overlap_height:, :]
    
    cv2.line(stitched_image, (0, stitch_y), (stitched_width, stitch_y), (0, 0, 255), 2)
    
    return stitched_image

# Example usage:
# Load two sample frames (replace these paths with your actual frame paths)
image1 = cv2.imread("Leaf10.png")
image2 = cv2.imread("Leaf30.png")

stitched = stitch_images(image1, image2)

cv2.imshow("Stitched Image", stitched)
print(stitched.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("LeafSegments/stitched_leaf.png", stitched)

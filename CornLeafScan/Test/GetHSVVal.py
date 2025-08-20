# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20


import cv2
import numpy as np

def resize_for_display(image, max_width=1000, max_height=800):
    """ Resize image while maintaining aspect ratio for display. """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def nothing(x):
    pass

# Load image
#image = cv2.imread("/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/Results/view_480.jpg")
image = cv2.imread("/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/Videos/frames/frame_320.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mean_saturation = np.mean(hsv[:, :, 1])
mean_value = np.mean(hsv[:, :, 2])

print(f"Mean Sat: {mean_saturation}")
print(f"Mean Val: {mean_value}")

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for H, S, and V (lower and upper bounds)
cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)  # Lower Hue
cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)  # Lower Saturation
cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)  # Lower Value
cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)  # Upper Hue
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)  # Upper Saturation
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)  # Upper Value

while True:
    # Get values from trackbars
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    # Create mask
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show results
    cv2.imshow("Mask", resize_for_display(mask))
    cv2.imshow("Filtered Result", resize_for_display(result))

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

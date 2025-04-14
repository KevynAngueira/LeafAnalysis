# Author: Kevyn Angueira Irizarry
# Created: 2025-03-25
# Last Modified: 2025-04-14

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
image = cv2.imread("/home/icicle/VSCode/LeafAnalysis/AdjusterTool2/LeafMedia/006/healthy/results/01/leafSegments/frame_1.jpg")
lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for L, a, and b (lower and upper bounds)
cv2.createTrackbar("LL", "Trackbars", 0, 255, nothing)  # Lower L (Lightness)
cv2.createTrackbar("La", "Trackbars", 0, 255, nothing)  # Lower a (Green-Red)
cv2.createTrackbar("Lb", "Trackbars", 0, 255, nothing)  # Lower b (Blue-Yellow)
cv2.createTrackbar("UL", "Trackbars", 255, 255, nothing)  # Upper L (Lightness)
cv2.createTrackbar("Ua", "Trackbars", 255, 255, nothing)  # Upper a (Green-Red)
cv2.createTrackbar("Ub", "Trackbars", 255, 255, nothing)  # Upper b (Blue-Yellow)

while True:
    # Get values from trackbars
    ll = cv2.getTrackbarPos("LL", "Trackbars")
    la = cv2.getTrackbarPos("La", "Trackbars")
    lb = cv2.getTrackbarPos("Lb", "Trackbars")
    ul = cv2.getTrackbarPos("UL", "Trackbars")
    ua = cv2.getTrackbarPos("Ua", "Trackbars")
    ub = cv2.getTrackbarPos("Ub", "Trackbars")

    # Create mask
    lower_bound = np.array([ll, la, lb]) 
    upper_bound = np.array([ul, ua, ub])
    mask = cv2.inRange(lab, lower_bound, upper_bound)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show results
    cv2.imshow("Mask", resize_for_display(mask))
    cv2.imshow("Filtered Result", resize_for_display(result))

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

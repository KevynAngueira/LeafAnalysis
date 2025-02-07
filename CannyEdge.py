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
image = cv2.imread("leaf_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for H, S, and V (lower and upper bounds)
cv2.createTrackbar("Lower", "Trackbars", 0, 255, nothing)  # Lower Hue
cv2.createTrackbar("Upper", "Trackbars", 0, 255, nothing)  # Lower Saturation
cv2.createTrackbar("Blur", "Trackbars", 5, 20, nothing)  # Lower Value

while True:
    # Get values from trackbars
    lower = cv2.getTrackbarPos("Lower", "Trackbars")
    upper = cv2.getTrackbarPos("Upper", "Trackbars")
    blur = cv2.getTrackbarPos("Blur", "Trackbars")
    
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, lower, upper)  # 100 and 200 are the lower and upper thresholds

    # Show results
    cv2.imshow("Edges", resize_for_display(edges))
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
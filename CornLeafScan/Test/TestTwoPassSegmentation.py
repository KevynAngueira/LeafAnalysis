# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np

from LeafScan.Misc.ResizeForDisplay import resize_for_display

def mean_channel(img, mask, channel_index):
    """Compute mean of a single Lab channel under mask"""
    channel = img[:, :, channel_index]
    return cv2.mean(channel, mask=mask)[0]

# Load image and convert to Lab
img = cv2.imread("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Step 1: loose mask in Lab (keeps all greenish stuff)
lower = np.array([0, 0, 123])     # wide open range
upper = np.array([255, 150, 255])
mask = cv2.inRange(lab, lower, upper)

# Step 2: apply mask to original image
masked_img = cv2.bitwise_and(img, img, mask=mask)
kernel = np.ones((3,3), np.uint8)
masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 3: find contours on grayscale masked image
gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 1)
sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
median = np.median(sharp)
low = int(max(0, 0.5 * median))
high = int(min(255, 1.5 * median))
edges = cv2.Canny(sharp, low, high)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
closed = cv2.dilate(edges, kernel, iterations=1)

# Step 5: Find contours
contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

all_contours = img.copy()
cv2.drawContours(all_contours, contours, -1, (0,0,255), 2)

# Step 6: filter contours by greenness (mean A-channel)
A_GREEN_THRESHOLD = 132
leaf_contours = []
background_holes = []


for i, c in enumerate(contours):
    perim = cv2.arcLength(c, True)
    if perim < 50:
        continue

    mask_c = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(mask_c, [c], -1, 255, -1)
    mean_A = mean_channel(lab, mask_c, 1)

    print(mean_A)
    if mean_A < A_GREEN_THRESHOLD:
        leaf_contours.append(c)
        
        cv2.imshow("Contour C", resize_for_display(mask_c))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if hierarchy[0][i][3] != -1:
            background_holes.append(c)


# Step A: Draw leaf contours
leaf_mask = np.zeros(mask.shape, np.uint8)
for c in leaf_contours:
    cv2.drawContours(leaf_mask, [c], -1, 255, -1)
    cv2.drawContours(leaf_mask, [c], -1, 255, 2)

# Step B: Remove holes
for c in background_holes:
    cv2.drawContours(leaf_mask, [c], -1, 0, -1)
        
# Apply mask to original image
isolated_leaf = cv2.bitwise_and(img, img, mask=leaf_mask)

# Save / show result
cv2.imwrite("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/isolated_leaf.png", isolated_leaf)

cv2.imshow("Isolated Leaf", resize_for_display(isolated_leaf))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Leaf Mask", resize_for_display(mask))
cv2.imshow("Leaf Result", resize_for_display(masked_img))
cv2.imshow("Contours", resize_for_display(all_contours))
cv2.imshow("Edges", resize_for_display(edges))
cv2.imshow("Out", resize_for_display(isolated_leaf))

cv2.waitKey(0)
cv2.destroyAllWindows()

# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from LeafScan.Misc.ResizeForDisplay import resize_for_display

def mean_channel(img, mask, channel_index):
    """Compute mean of a single Lab channel under mask"""
    channel = img[:, :, channel_index]
    return cv2.mean(channel, mask=mask)[0]

# Load image and convert to Lab
img = cv2.imread("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Step 1: loose mask in Lab (keeps all greenish stuff)
lower = np.array([0, 0, 126])     # wide open range
upper = np.array([255, 140, 255])
mask = cv2.inRange(lab, lower, upper)

# Step 2: apply mask to original image
masked_img = cv2.bitwise_and(img, img, mask=mask)
kernel = np.ones((3,3), np.uint8)
masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 3: find contours on grayscale masked image
gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 1)
sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
median = np.median(sharp[sharp>0])
print(median)
low = int(max(0, 0.5 * median))
high = int(min(255, 1.5 * median))
edges = cv2.Canny(sharp, low, high)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
closed = cv2.dilate(edges, kernel, iterations=2)

# Suppose 'mask' is your binary mask (0/255)
h, w = mask.shape[:2]
# Draw a rectangle along the border
cv2.rectangle(closed, (0, 0), (w-1, h-1), 255, 2)

# Step 5: Find contours
contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

all_contours = img.copy()
cv2.drawContours(all_contours, contours, -1, (0,0,255), 2)

print(len(contours))

contour_stats = []
for i, c in enumerate(contours):
    parent = hierarchy[0][i][3]
    if parent == -1:
       continue
    
    perim = cv2.arcLength(c, True)
    if perim < 10:
        continue
    
    mask_c = np.zeros(mask.shape, np.uint8)    
    cv2.drawContours(mask_c, [c], -1, 255, -1)

    mean_A = mean_channel(lab, mask_c, 1)
    mean_B = mean_channel(lab, mask_c, 2)

    temp = masked_img.copy()
    cv2.drawContours(temp, [c], -1, 255, 2)

    #print(mean_A, mean_B)
    #cv2.imshow("Contour C", resize_for_display(temp))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    parent = hierarchy[0][i][3]
    contour_stats.append([mean_A, mean_B, c, parent])


# === Build feature array from contours ===
AB = np.array([[x[0], x[1]] for x in contour_stats])

'''
# --- STRATEGY 1: inject background anchor samples ---
h, w = lab.shape[:2]

# Sample some pixels from left and right edges (assume background)
edge_samples = []
for y in np.linspace(0, h-1, 10, dtype=int):  # 10 evenly spaced rows
    edge_samples.append(lab[y, 2, 1:3])       # near left edge
    edge_samples.append(lab[y, w-3, 1:3])     # near right edge

edge_samples = np.array(edge_samples, dtype=np.float32)

# Stack contour features with background anchors
AB_with_bg = np.vstack([AB, edge_samples])

# Run KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(AB_with_bg)

# Only keep labels for the actual contours
labels = kmeans.labels_[:len(AB)]
'''

kmeans = KMeans(n_clusters=2, random_state=0).fit(AB)
labels = kmeans.labels_

try:
    score = silhouette_score(AB, labels)
except ValueError:
    score = 0  # not enough samples for silhouette

if score < 0.1:  # low separation → clusters not meaningful
    print("⚠️ Clusters too close, treating all as leaf")
    labels[:] = 0  # force all into one cluster

leaf_cluster_idx = np.argmin(kmeans.cluster_centers_[:,0])
print(kmeans.cluster_centers_[:,0])
print(kmeans.cluster_centers_[:,1])

print(leaf_cluster_idx)
leaf_contours = []
background_holes = []

for i, stat in enumerate(contour_stats):
    c = stat[2]
    parent = stat[3]
    if labels[i] == leaf_cluster_idx:
        print(f"Leaf: {stat[0]}, {stat[1]}")
        leaf_contours.append(c)
    else:
        print(f"Background: {stat[0]}, {stat[1]}")
        background_holes.append(c)

# Step A: combine all leaf contours into a single mask
leaf_mask = np.zeros(mask.shape, np.uint8)
cv2.fillPoly(leaf_mask, leaf_contours, 255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
leaf_mask_closed = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

# Step B: extract one clean outer contour from that union
contours_merged, _ = cv2.findContours(leaf_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_mask_final = np.zeros(mask.shape, np.uint8)

for cnt in contours_merged:
    # Smooth/approximate each contour
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    merged_poly = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(leaf_mask_final, [merged_poly], -1, 255, -1)

cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
cv2.imshow("Closed Contours", resize_for_display(leaf_mask_closed))
cv2.imshow("New Contours", resize_for_display(leaf_mask_final))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step C: subtract the background holes
for c in background_holes:
    cv2.drawContours(leaf_mask_final, [c], -1, 0, -1)

# Final isolated leaf
isolated_leaf = cv2.bitwise_and(img, img, mask=leaf_mask_final)

print(len(leaf_contours))
print(len(background_holes))

# Save / show result
cv2.imwrite("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/isolated_leaf.png", isolated_leaf)

cv2.imshow("Isolated Leaf", resize_for_display(isolated_leaf))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Leaf Mask", resize_for_display(mask))
cv2.imshow("Leaf Result", resize_for_display(masked_img))
cv2.imshow("Contours", resize_for_display(all_contours))
cv2.imshow("Edges", resize_for_display(edges))
cv2.imshow("Sharpen", resize_for_display(sharp))
cv2.imshow("Out", resize_for_display(isolated_leaf))

cv2.waitKey(0)
cv2.destroyAllWindows()

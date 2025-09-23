# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np
from sklearn.cluster import KMeans
import time

# --- Config ---
input_image_path = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png"
output_kmeans = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/kmeans_result.png"
output_leaf = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/leaf_segment.png"

# --- Load and convert ---
img = cv2.imread(input_image_path)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
h, w = lab.shape[:2]

# --- Use only A and B channels for clustering ---
pixels_ab = lab.reshape((-1, 3))[:, 1:]  # (num_pixels, 2)

# ------------------- K-MEANS -------------------
start_time = time.time()
kmeans = KMeans(n_clusters=4, random_state=42).fit(pixels_ab)
kmeans_time = time.time() - start_time

labels_kmeans = kmeans.labels_.reshape(h, w)
centroids = kmeans.cluster_centers_

# --- Heuristic: assign clusters ---
leaf_cluster    = np.argmin(centroids[:, 0])   # min a* = greenest
back_cluster    = np.argmin(centroids[:, 1])   # min b* = bluest
border_cluster  = np.argmax(centroids[:, 0])   # max a* = pinkest

# Assign 4th cluster (whichever is left)
all_clusters = {leaf_cluster, back_cluster, border_cluster}
remaining = set(range(4)) - all_clusters
if remaining:
    leftover_cluster = remaining.pop()
    # Assign leftover to whichever centroid it’s closest to
    dists = {
        "leaf": np.linalg.norm(centroids[leftover_cluster] - centroids[leaf_cluster]),
        "back": np.linalg.norm(centroids[leftover_cluster] - centroids[back_cluster]),
        "border": np.linalg.norm(centroids[leftover_cluster] - centroids[border_cluster]),
    }
    closest = min(dists, key=dists.get)
    if closest == "leaf":
        leaf_cluster = [leaf_cluster, leftover_cluster]
    elif closest == "back":
        back_cluster = [back_cluster, leftover_cluster]
    else:
        border_cluster = [border_cluster, leftover_cluster]

# --- Visualization (use cv2.mean for natural coloring) ---
kmeans_colors = np.zeros_like(img)
for i in range(4):
    mask = (labels_kmeans == i).astype(np.uint8)
    if np.any(mask):
        mean_color = cv2.mean(img, mask*255)[:3]
        kmeans_colors[mask == 1] = mean_color

# --- Create final leaf mask ---
if isinstance(leaf_cluster, list):
    leaf_mask = np.isin(labels_kmeans, leaf_cluster).astype(np.uint8) * 255
else:
    leaf_mask = (labels_kmeans == leaf_cluster).astype(np.uint8) * 255

# Apply mask to original image
leaf_segment = cv2.bitwise_and(img, img, mask=leaf_mask)

# --- SHOW RESULTS ---
cv2.imshow("Original", img)
cv2.imshow("K-means Visualization", kmeans_colors)
cv2.imshow("Leaf Mask", leaf_mask)
cv2.imshow("Segmented Leaf", leaf_segment)
print(f"K-means time: {kmeans_time:.2f} s")
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Save results ---
cv2.imwrite(output_kmeans, kmeans_colors)
cv2.imwrite(output_leaf, leaf_segment)

# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import time

# --- Config ---
input_image_path = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png"
output_kmeans = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/kmeans_result.png"
output_dbscan = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/dbscan_result.png"

dbscan_subsample = 0.1   # fraction of pixels to use for DBSCAN (0.1 = 10%)
dbscan_eps = 10           # spatial/color distance for DBSCAN
dbscan_min_samples = 10

# --- Load and convert ---
img = cv2.imread(input_image_path)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
h, w = lab.shape[:2]

# Flatten pixels for clustering (use A and B channels)
pixels = lab.reshape((-1,3))[:,1:]  # shape: (num_pixels, 2)

# ------------------- K-MEANS -------------------
start_time = time.time()
kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
kmeans_time = time.time() - start_time

labels_kmeans = kmeans.labels_.reshape(h, w)

# Map labels to colors for visualization
kmeans_colors = np.zeros_like(img)
for i in range(3):
    mask = (labels_kmeans == i)
    kmeans_colors[mask] = cv2.mean(img, mask.astype(np.uint8)*255)[:3]

# --- DBSCAN ---
start = time.time()
# Subsample pixels
n_samples = int(len(pixels) * dbscan_subsample)
subsample_idx = np.random.choice(len(pixels), n_samples, replace=False)
subsample_pixels = pixels[subsample_idx]

dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
dbscan.fit(subsample_pixels)

# Assign cluster labels back to all pixels based on nearest subsample
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(subsample_pixels)
distances, indices = nn.kneighbors(pixels)
dbscan_labels = dbscan.labels_[indices.flatten()].reshape(h, w)
dbscan_time = time.time() - start

# Map DBSCAN clusters to colors
unique_labels = np.unique(dbscan_labels)
dbscan_result = np.zeros_like(img)
for label in unique_labels:
    if label == -1:
        color = np.array([0, 0, 0], dtype=np.uint8)  # noise -> black
    else:
        # Get the pixels corresponding to this cluster in the subsample
        cluster_pixels = subsample_pixels[dbscan.labels_ == label]
        if cluster_pixels.shape[1] == 3:
            color = np.mean(cluster_pixels, axis=0).astype(np.uint8)
        else:
            # fallback in case of shape issues
            color = np.array([0, 0, 0], dtype=np.uint8)
    dbscan_result[dbscan_labels == label] = color
    
# --- SHOW RESULTS ---
cv2.imshow("Original", img)
cv2.imshow("K-means", kmeans_)
cv2.imshow("DBSCAN", dbscan_result)
print(f"K-means time: {kmeans_time:.2f} s")
print(f"DBSCAN time: {dbscan_time:.2f} s")
cv2.waitKey(0)
cv2.destroyAllWindows()

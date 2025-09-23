import cv2
import numpy as np
from sklearn.cluster import KMeans

# --- Config ---
input_image_path = "/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png"
tile_size = 64   # tile size (height and width)
clusters = 4     # run k-means with 2 clusters per tile

# --- Load image and convert to LAB ---
img = cv2.imread(input_image_path)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
h, w = lab.shape[:2]

# Prepare output image
segmented = np.zeros_like(img)

# --- Process tiles ---
for y in range(0, h, tile_size):
    for x in range(0, w, tile_size):
        # Extract tile
        tile = lab[y:y+tile_size, x:x+tile_size]
        th, tw = tile.shape[:2]

        # Flatten to (N,2) using only A and B channels
        pixels = tile.reshape((-1, 3))[:, 1:]  

        # Run k-means (2 clusters)
        kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit(pixels)
        labels = kmeans.labels_.reshape(th, tw)

        # Map clusters to colors (for now, just use average BGR of each cluster)
        tile_result = np.zeros((th, tw, 3), dtype=np.uint8)
        for i in range(clusters):
            mask = (labels == i)
            if np.any(mask):
                # Compute average color from original BGR image
                mean_color = cv2.mean(img[y:y+th, x:x+tw], mask.astype(np.uint8) * 255)[:3]
                tile_result[mask] = mean_color

        # Place segmented tile into output
        segmented[y:y+th, x:x+tw] = tile_result

# --- Show results ---
cv2.imshow("Original", img)
cv2.imshow("Tiled K-means (2 clusters)", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np

from sklearn.cluster import KMeans

from Misc.ResizeForDisplay import resize_for_display

from Configs import LeafExtractorConfig

from BinaryMask import LABMask

import time
from contextlib import contextmanager

@contextmanager
def timer(name="Process"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] {end - start:.4f} s")


class KmeansLeafExtractor:
    def __init__(self, config: LeafExtractorConfig=None, clusters = 7, padding = 20):
        if config is None:
            config = LeafExtractorConfig()
        self.__dict__.update(vars(config))

        self.clusters = clusters
        self.padding = padding

        self.leafMask = LABMask(config.leaf_bounds)


    def _addAnchorPoints(self, image, mask, square_size=15, border_pad = 5, back_pad=30):
        
        h, w = image.shape[:2]
        s = square_size

        corner_coords = np.array([
            (0,0),
            (w - s, 0),
            (0, h - s),
            (w - s, h - s)
        ])

        pad_array = np.array([
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1)
        ])


        border_coords = corner_coords + (border_pad * pad_array)
        back_coords = corner_coords + (back_pad * pad_array)

        all_anchors = np.concatenate((border_coords, back_coords), axis=0)

        anchored_mask = mask.copy()
        for (x, y) in all_anchors:
            cv2.rectangle(anchored_mask, (x, y), (x + s, y + s), 255, -1)

        anchored_image = cv2.bitwise_and(image, image, mask=anchored_mask)
        
        return anchored_image, anchored_mask


    def _imagePreprocessing(self, image):
        
        leaf_result, leaf_mask = self.leafMask.applyMask(image)

        anchored_image, anchored_mask = self._addAnchorPoints(image, leaf_mask)

        lab = cv2.cvtColor(anchored_image, cv2.COLOR_BGR2LAB)
        
        return lab

    def _clusterImage(self, image, subsample=0.25):
        h, w = image.shape[:2]
        pixels = image.reshape((-1, 3))
        L, A, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

        # Ignore empty (masked) pixels
        valid_mask = L > 0
        pixels_ab = np.column_stack((A[valid_mask], B[valid_mask]))

        # --- Subsample ---
        if subsample < 1.0:
            n_samples = int(len(pixels_ab) * subsample)
            idx = np.random.choice(len(pixels_ab), n_samples, replace=False)
            pixels_ab_sample = pixels_ab[idx]
        else:
            pixels_ab_sample = pixels_ab

        # Fit only on sample
        kmeans = KMeans(n_clusters=self.clusters, random_state=42).fit(pixels_ab_sample)
        centroids = kmeans.cluster_centers_

        # Predict on *all* valid pixels
        labels_valid = kmeans.predict(pixels_ab)

        # Reconstruct full label map
        labels = np.full((h * w,), -1, dtype=np.int32)
        labels[valid_mask] = labels_valid
        labels = labels.reshape(h, w)

        return labels, centroids
    
    def _checkLeafExists(self, centroids):
        """
        Check if a leaf cluster is present.
        Leaf cluster characteristics:
        - A* is low (green)
        - B* is high (yellow-blue axis)
        
        Method:
        - Take the cluster with lowest A* (greenest)
        - Compare its A* and B* relative to other clusters
        - Return True if it satisfies thresholds
        """
        a_vals = centroids[:, 0]
        b_vals = centroids[:, 1]

        greenest_idx = np.argmin(a_vals)
        a_greenest = a_vals[greenest_idx]
        b_greenest = b_vals[greenest_idx]

        # Heuristic: A must be sufficiently below median, B must be sufficiently above median
        a_thresh = np.median(a_vals)
        b_thresh = np.median(b_vals)

        if (a_greenest < a_thresh) and (b_greenest > b_thresh):
            return True
        else:
            return False

    def _assignClusters(self, image, labels, centroids, max_iter=5):
        """
        Assign clusters using color heuristics with iterative refinement for leaf.
        - Leaf: greenest (min A*)
        - Back: bluest (min B*)
        - Border: pinkest (max A*)
        
        Leftover clusters are assigned based on channel-specific heuristic.
        Iteratively recompute the leaf centroid to catch intermediate greens.
        """

        # --- Initial assignment ---
        leaf_cluster   = np.argmin(centroids[:, 0])  # min a* = greenest
        back_cluster   = np.argmin(centroids[:, 1])  # min b* = bluest
        border_cluster = np.argmax(centroids[:, 0])  # max a* = pinkest

        # Always store as lists for consistency
        cluster_groups = {
            "leaf": [leaf_cluster],
            "back": [back_cluster],
            "border": [border_cluster],
        }

        remaining = set(range(self.clusters)) - {leaf_cluster, back_cluster, border_cluster}

        for iteration in range(max_iter):
            changed = False
            new_leaf_group = []

            # Compute current leaf centroid as mean of assigned subclusters
            leaf_centroid = centroids[cluster_groups["leaf"]].mean(axis=0)
            #print(centroids[cluster_groups["leaf"]])

            for leftover in remaining:
                a_leftover, b_leftover = centroids[leftover, 0], centroids[leftover, 1]

                # Compare A channel with border vs leaf
                a_border = centroids[border_cluster, 0]
                dist_a_border = abs(a_leftover - a_border)
                dist_a_leaf   = abs(a_leftover - leaf_centroid[0])

                # Compare B channel with back vs leaf
                b_back = centroids[back_cluster, 1]
                dist_b_back = abs(b_leftover - b_back)
                dist_b_leaf = abs(b_leftover - leaf_centroid[1])

                # Apply heuristic
                if 1.2 * dist_a_border < dist_a_leaf:
                    closest = "border"
                elif 1.4 * dist_b_back < dist_b_leaf:
                    closest = "back"
                else:
                    closest = "leaf"

                if closest == "leaf":
                    new_leaf_group.append(leftover)
                    if leftover not in cluster_groups["leaf"]:
                        changed = True
                else:
                    cluster_groups[closest].append(leftover)

            # Update leaf cluster
            cluster_groups["leaf"] = list(set(cluster_groups["leaf"]) | set(new_leaf_group))

            if not changed:
                break  # stop iterating when leaf group stabilizes
        
        return cluster_groups["leaf"], cluster_groups["back"], cluster_groups["border"]


    def _segmentLeaf(self, image, leaf_cluster, labels):
        
        # Create leaf mask
        leaf_cluster = list(leaf_cluster)
        leaf_mask = np.isin(labels, leaf_cluster).astype(np.uint8) * 255
        
        # Apply mask to original image
        leaf_result = cv2.bitwise_and(image, image, mask=leaf_mask)

        return leaf_result, leaf_mask

    def _visualizeClusters(self, image, labels):

        # Vizualize Clusters
        visualized_clusters = np.zeros_like(image)
        for i in range(self.clusters):
            mask = (labels == i).astype(np.uint8)
            if np.any(mask):
                mean_color = cv2.mean(image, mask*255)[:3]
                visualized_clusters[mask == 1] = mean_color
        
        return visualized_clusters

    def _cropBorder(self, image, padding):
        h, w = image.shape[:2]
        cropped = image[padding: h-padding, padding:w-padding]
        resized = cv2.resize(cropped, self.target_dimensions)
        return resized

    def Extract(self, image, display=False, deep_display=False):
        
        with timer("Preprocessing"):
            preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        #preprocessed = self._imagePreprocessing(image)

        
        with timer("KMeans Clustering"):
            labels, centroids = self._clusterImage(preprocessed)
        
        if self._checkLeafExists(centroids):
            with timer("Assign Clusters"):
                leaf_cluster, back_cluster, border_cluster = self._assignClusters(image, labels, centroids)
            
            with timer("Segement"):
                leaf_result, leaf_mask = self._segmentLeaf(image, leaf_cluster, labels)
            
            with timer("Crop"):
                leaf_result = self._cropBorder(leaf_result, self.padding)
                leaf_mask = self._cropBorder(leaf_mask, self.padding)

            if display:
                visualized_clusters = self._visualizeClusters(image, labels)
                cv2.imshow("Visualized", resize_for_display(visualized_clusters))
        else:
            leaf_result = np.zeros(image.shape, np.uint8)
            leaf_mask = np.zeros(image.shape, np.uint8)


        if display:
            cv2.imshow("Leaf Mask", resize_for_display(leaf_mask))
            cv2.imshow("Leaf Result", resize_for_display(leaf_result))
            
        return leaf_result, leaf_mask

# Author: Kevyn Angueira Irizarry
# Created: 2025-09-22
# Last Modified: 2025-09-22

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from Misc.ResizeForDisplay import resize_for_display

from Configs import LeafExtractorConfig

from BinaryMask import LABMask

class ContourLeafExtractor:
    def __init__(self, config: LeafExtractorConfig=None):
        if config is None:
            config = LeafExtractorConfig()
        self.__dict__.update(vars(config))

        self.leafMask = LABMask(config.leaf_bounds)
    
    def _blurAndSharpen(self, channel):
        blurred = cv2.GaussianBlur(channel, self.blur_kernel, 1)
        sharpened = cv2.addWeighted(channel, self.sharpen_weight[0], blurred, self.sharpen_weight[1], 0)
        return sharpened

    def _imagePreprocessing(self, image):
        """
        Applying preprocessing to the image.
        """

        # Setup
        leaf_result, leaf_mask = self.leafMask.applyMask(image)
        L, A, B = cv2.split(leaf_result)

        # Blur and Sharpen the LAB channels
        L_sharp = self._blurAndSharpen(L)
        A_sharp = self._blurAndSharpen(A)
        B_sharp = self._blurAndSharpen(B)

        # Merge back into a single image
        preprocessed = cv2.merge([L_sharp, A_sharp, B_sharp])
        return preprocessed


    def filter_wrapper_contours(self, contours, hierarchy, area_ratio_thresh=0.65):
        """
        Removes wrapper contours and returns filtered contours + simplified hierarchy.
        Wrapper contours are those that wrap around child contours but are not meaningfully distinct. 
        In our case checking for wrapped around a leaf contour with added background elements.
        """
        keep = [True] * len(contours)
        parent_flags = [-1] * len(contours)  # -1 = no parent, else parent index

        for idx, h in enumerate(hierarchy[0]):
            parent_idx = idx
            first_child = h[2]  # index of first child
            parent_flags[idx] = h[3]  # store original parent index (or -1)

            if first_child != -1:
                parent_area = cv2.contourArea(contours[parent_idx])
                child_area_sum = 0
                child_idx = first_child
                while child_idx != -1:
                    child_area_sum += cv2.contourArea(contours[child_idx])
                    child_idx = hierarchy[0][child_idx][0]  # next sibling

                area_ratio = child_area_sum / (parent_area + 1e-6)
                if area_ratio > area_ratio_thresh:
                    keep[parent_idx] = False
                    
                """
                print(area_ratio)
                temp = np.zeros((135, 660), np.uint8)    
                cv2.drawContours(temp, [contours[parent_idx]], -1, 255, -1)
                cv2.imshow("Testing C", resize_for_display(temp))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """

        # Filtered outputs
        filtered_contours = [c for c, k in zip(contours, keep) if k]
        filtered_hierarchy = [p for p, k in zip(parent_flags, keep) if k]

        return filtered_contours, filtered_hierarchy

    def _addBackgroundAnchors(self, edges):
        """
        Adds anchor point rectangles at the inside the light blue backpiece and the pink border. 
        These anchor points help anchor the future clustering classification providing strong identifiers for the background and pink border
        """

        # ==== Light Blue Backpiece Anchor Points ====
        anchored = edges.copy()
        h, w = anchored.shape[:2]
        pad = self.padding  # number of pixels padded around all sides

        # Define usable region (inside padding)
        usable_w = w - 2 * pad
        usable_h = h - 2 * pad

        # Parameters (relative to usable region)
        margin_x = int(0.02 * usable_w)   # horizontal margin from usable edge
        margin_y = int(0.15 * usable_h)   # vertical margin from top/bottom
        grid_rows = 4
        square_w = int(0.02 * usable_w)
        square_h = int((usable_h - 2 * margin_y) / grid_rows) - 5
        v_spacing = 5

        # Left anchor X positions (inside left usable region)
        left_x_positions = [
            pad + margin_x,
            pad + margin_x + square_w + 5
        ]

        # Right anchor X positions (inside right usable region)
        right_x_positions = [
            w - pad - margin_x - 2*square_w - 5,
            w - pad - margin_x - square_w
        ]

        # Y positions for 4 rows (inside usable region)
        y_positions = [
            pad + margin_y + (v_spacing//2) + i*(square_h + v_spacing)
            for i in range(grid_rows)
        ]

        # Draw squares
        for y in y_positions:
            for x in left_x_positions:
                cv2.rectangle(anchored, (x, y), (x + square_w, y + square_h), 255, -1)
            for x in right_x_positions:
                cv2.rectangle(anchored, (x, y), (x + square_w, y + square_h), 255, -1)

        # ==== Pink Border Anchor Points ====
        corner_size = int(0.05 * min(w, h))  # 3% of shorter side
        pad = 10
        corner_coords = np.array([
            (0, 0),                              # top-left
            (w - corner_size, 0),                # top-right
            (0, h - corner_size),                # bottom-left
            (w - corner_size, h - corner_size)   # bottom-right
        ])
        padding = np.array([
            (pad, pad),                              # top-left
            (-pad, pad),                # top-right
            (pad, -pad),                # bottom-left
            (-pad, -pad)   # bottom-right
        ])
        corner_coords = corner_coords + padding

        for (x, y) in corner_coords:
            cv2.rectangle(anchored, (x, y), (x + corner_size, y + corner_size), 255, -1)

        return anchored

    def _getEdges(self, channel):
        # Detect image edges, dynamically scaled to image brightness
        median = np.median(channel[channel > 0])
        canny_low = int(max(0, self.canny_coef[0] * median))
        canny_high = int(min(255, self.canny_coef[1] * median))

        # Invert the mask to gurantee complete contours
        masked_img = np.full_like(channel, 255)
        masked_img[channel>0] = channel[channel>0]
        
        edges = cv2.Canny(masked_img, canny_low, canny_high)

        return edges

    def  _getContours(self, preprocessed, display = False):
        """
        Extract contours from the image, dynamically scaled to image brightness
        """
        
        _, A_channel, B_channel = cv2.split(preprocessed)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.ellipse_kernel)
        A_morph = cv2.morphologyEx(A_channel, cv2.MORPH_CLOSE, morph_kernel)
        B_morph = cv2.morphologyEx(B_channel, cv2.MORPH_CLOSE, morph_kernel)
         
        A_edges = self._getEdges(A_morph)
        B_edges = self._getEdges(B_morph)

        all_edges = cv2.bitwise_or(A_edges, B_edges)

        closed = cv2.morphologyEx(all_edges, cv2.MORPH_CLOSE, morph_kernel, iterations=self.morph_iterations) 

        # Add a rectangle box to the border of the image (helps close open contours)
        h, w = preprocessed.shape[:2]
        cv2.rectangle(closed, (0, 0), (w-1, h-1), 255, 2)

        # Add background anchor points for classification calibration
        anchored = self._addBackgroundAnchors(closed)

        # Turn image to closed gradients
        blob_edges = cv2.morphologyEx(anchored, cv2.MORPH_GRADIENT, np.ones((self.morph_kernel), np.uint8))
        
        # Find contours
        contours, hierarchy = cv2.findContours(blob_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter wrapper contours
        filtered_contours, filtered_hierarchy = self.filter_wrapper_contours(contours, hierarchy)

        drawn_contours = np.zeros(preprocessed.shape[:2], np.uint8)
        cv2.drawContours(drawn_contours, filtered_contours, -1, 255, 2)

        """
        cv2.imshow("Edges", resize_for_display(all_edges))
        cv2.imshow("Closed", resize_for_display(closed))
        cv2.imshow("Blob Edges", resize_for_display(blob_edges))
        cv2.imshow("All Contours", resize_for_display(drawn_contours))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        return filtered_contours, filtered_hierarchy

    def _mean_channel(self, img, mask, channel_index):
        """Compute mean of a single Lab channel under mask"""
        channel = img[:, :, channel_index]
        return cv2.mean(channel, mask=mask)[0]

    def _getContourStats(self, original, contours, hierarchy, deep_display=False):
        """
        Arrange contours based on mean A and B color channels for future classification
        """
        
        leaf_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        area_threshold = 0.9 * (original.shape[0] - 2*self.padding) * (original.shape[1] - 2*self.padding)

        # Capture A and B channel stats for future contour classification
        contour_stats = []
        for i, c in enumerate(contours):
            
            # Eliminate parent contour (encompasses the full image)
            parent = hierarchy[i]
            area = cv2.contourArea(c)
            if parent == -1 or area > area_threshold:
                continue
            
            # Eliminate small noise contours
            perim = cv2.arcLength(c, True)
            if perim < 20:
                continue
            
            # Isolate contour
            mask_c = np.zeros(original.shape[:2], np.uint8)    
            cv2.drawContours(mask_c, [c], -1, 255, -1)

            # Calculate contour stats
            mean_A = self._mean_channel(leaf_lab, mask_c, 1)
            mean_B = self._mean_channel(leaf_lab, mask_c, 2)

            contour_stats.append([mean_A, mean_B, c])
            
            if deep_display:
                print(mean_A, mean_B)
                temp = original.copy()
                cv2.drawContours(temp, [c], -1, (0,255,255), 2)
                
                cv2.imshow("Contour C", resize_for_display(temp))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return contour_stats

    def _classifyContours(self, contour_stats, deep_display=False):
        """
        Classify contours between leaf contours and background contours
        """

        # Classify contours into leaf and background clusters
        AB = np.array([stat[:2] for stat in contour_stats])
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=0).fit(AB)
        labels = kmeans.labels_

        # Fuse clusters if too similar
        try:
            score = silhouette_score(AB, labels)
        except ValueError:
            score = 0  
        
        if score < 0.1:  # low separation → clusters not meaningful
            print("⚠️ Clusters too close, treating all as leaf")
            labels[:] = 0          
        
        if np.all(kmeans.cluster_centers_[:,0] > 140):
            print("No leaf cluster")
            leaf_cluster_idx = 0
            labels[:] = 1
        else:
            # Detect leaf cluster based on mean "greeness"
            leaf_cluster_idx = np.argmin(kmeans.cluster_centers_[:,0])

        leaf_contours = []
        background_contours = []
        for i, stat in enumerate(contour_stats):
            c = stat[2]
            if labels[i] == leaf_cluster_idx:
                if deep_display: print(f"Leaf: {stat[0]}, {stat[1]}")
                leaf_contours.append(c)
            else:
                if deep_display: print(f"Background: {stat[0]}, {stat[1]}")
                background_contours.append(c)
        print(kmeans.cluster_centers_[:,0])
        print(kmeans.cluster_centers_[:,1])

        return leaf_contours, background_contours

    def _contoursToLeaf(self, original, leaf_contours, background_contours):

        #Combine all leaf contours into a single mask
        leaf_mask = np.zeros(original.shape[:2], np.uint8)
        cv2.fillPoly(leaf_mask, leaf_contours, 255)

        # Merge adjacent contours into larger leaf contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.ellipse_kernel)
        leaf_mask_closed = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

        # Detect new outer contours for merged leaf segments
        contours_merged, _ = cv2.findContours(leaf_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Smooth and fill outer contours into final clean polygonal shape
        leaf_mask_final = np.ones(original.shape[:2], np.uint8)*0
        for cnt in contours_merged:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            merged_poly = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(leaf_mask_final, [merged_poly], -1, 255, -1)

        # Substract background contours/holes
        #for c in background_contours:
        #    cv2.drawContours(leaf_mask_final, [c], -1, 0, -1)

        # Final isolated leaf
        leaf_result = cv2.bitwise_and(original, original, mask=leaf_mask_final)

        return leaf_result, leaf_mask_final

    def _cropBorder(self, img, padding):
        h, w = img.shape[:2]
        cropped = img[padding: h-padding, padding:w-padding]
        resized = cv2.resize(cropped, self.target_dimensions)
        return resized

    def Extract(self, image, display=False, deep_display=False):
        preprocessed = self._imagePreprocessing(image)
        contours, hierarchy = self._getContours(preprocessed)
        contour_stats = self._getContourStats(image, contours, hierarchy, deep_display=deep_display)
        leaf_contours, background_contours = self._classifyContours(contour_stats, deep_display=deep_display)
        leaf_result, leaf_mask = self._contoursToLeaf(image, leaf_contours, background_contours)

        final_result = self._cropBorder(leaf_result, self.padding)
        final_mask = self._cropBorder(leaf_mask, self.padding)

        if display:
            drawn_contours = image.copy()
            cv2.drawContours(drawn_contours, contours, -1, (0, 255, 255), 0)

            print("====== Leaf Extractor ======")
            print("Total Contours: ", len(contour_stats))
            print("Leaf Contours: ", len(leaf_contours))
            print("Background Contours", len(background_contours))
            print("====== Leaf Extractor ======")

            cv2.imshow("Original", resize_for_display(image))
            cv2.imshow("Preprocessed", resize_for_display(preprocessed))
            cv2.imshow("Contours", resize_for_display(drawn_contours))
            cv2.imshow("Leaf Result", resize_for_display(final_result))
            cv2.imshow("Leaf Mask", resize_for_display(final_mask))

        return final_result, final_mask

if __name__ == "__main__":
    original = cv2.imread("/home/icicle/VSCode/LeafAnalysis/CornLeafScan/TestLab/Tool/View Window_screenshot_26.08.20252.png")
   
    
    leafExtractor = DynamicLeafExtractor()

    isolated_leaf = leafExtractor.Extract(original, display=True)

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class LeafSegmentStitcher:
    def __init__(self, similarity_threshold=0.7):
        self.last_saved_frame = None
        self.segment_count = 0
        self.similarity_threshold = similarity_threshold

    def process_frame(self, frame):
        """ Check if the new frame is different enough to be saved. """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.last_saved_frame is None:
            self.save_frame(frame)
            return

        # Resize to match previous frame (if needed)
        prev_gray = cv2.resize(self.last_saved_frame, (gray_frame.shape[1], gray_frame.shape[0]))

        # Compute SSIM similarity
        similarity = ssim(prev_gray, gray_frame)

        print(f"SSIM Similarity: {similarity:.2f}")

        # Save frame if similarity is below threshold (indicating a new leaf segment)
        if similarity < self.similarity_threshold:
            self.save_frame(frame)

    def save_frame(self, frame):
        """ Save the frame as a new leaf segment. """
        filename = f"LeafSegments/leaf_segment_{self.segment_count}.png"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved new leaf segment: {filename}")

        # Store the new frame for future comparisons
        #self.last_saved_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_saved_frame = frame
        self.segment_count += 1

    def check_overlap(self, frame, threshold=0.1):
        
        if self.last_saved_frame is None:
            self.save_frame(frame)
            return True
        
        # Initialize frames
        prev_frame = self.last_saved_frame
        new_frame = frame
        
        # Initialize ORB detector
        orb = cv2.ORB_create()
        
        # Detect keypoints and descriptors in both frames
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(new_frame, None)

        # Check if descriptors were found
        if des1 is None or des2 is None:
            print("No descriptors found in one or both frames.")
            return False  # No descriptors, consider as overlap
        
        # Ensure descriptors are the same type (CV_8U)
        if des1.dtype != des2.dtype:
            print("Descriptors have different types, converting them to CV_8U.")
            des1 = np.uint8(des1)
            des2 = np.uint8(des2)
        
        # Use a Brute-Force Matcher to find matches between descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort the matches based on their distances
        matches = sorted(matches, key = lambda x:x.distance)
        
        # If the number of good matches is above a certain threshold, we calculate the homography
        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Compute homography using RANSAC to deal with outliers
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Warp the previous frame to align with the new one
            h, w = new_frame.shape
            prev_frame_warped = cv2.warpPerspective(prev_frame, M, (w, h))
            
            # Calculate the overlap (where both images have non-zero pixel values)
            overlap = np.sum(prev_frame_warped > 0) / np.sum(new_frame > 0)
            
            if overlap < threshold:
                self.save_frame(frame)
                return True  # There's no significant overlap; save the new frame
            else:
                return False  # There's overlap; do not save the new frame
        else:
            return False  # Not enough matches; treat as overlap
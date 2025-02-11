import cv2
import numpy as np

class LeafStitcher:
    def __init__(self):
        self.frames = []
        self.stitched_image = None

    def process_video(self, video_path):
        """ Process the video and stitch frames together. """
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                self.frames.append(frame)
                cv2.imwrite(f"Leaf{frame_count}.png", frame)
            frame_count += 1

        cap.release()
        self.stitch_images()

    def stitch_images(self):
        """ Stitch all frames together vertically using template matching. """
        if len(self.frames) < 2:
            print("Not enough frames to stitch.")
            return

        # Initialize the stitched image with the first frame
        self.stitched_image = self.frames[0]
        stitched_width = self.stitched_image.shape[1]
        stitched_height = self.stitched_image.shape[0]

        # Process frames and dynamically adjust canvas height based on overlap
        for i in range(1, len(self.frames)):
            frame = self.frames[i]
            
            # Convert both the stitched image and the current frame to grayscale
            gray_stitched = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Match the template (stitched image) in the current frame
            result = cv2.matchTemplate(gray_frame, gray_stitched, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            # Calculate the position of the template in the current frame (best match)
            top_left = max_loc
            bottom_right = (top_left[0] + stitched_width, top_left[1] + stitched_height)

            # Calculate the overlap region (vertical overlap based on Y-coordinate)
            overlap_height = top_left[1]  # The distance from the top of the frame to the match location
            
            # Determine how much new height to add based on the overlap
            new_height = frame.shape[0] - overlap_height

            # Ensure new_height is at least 1 (avoid zero height)
            if new_height < 1:
                new_height = 1

            # Check if the new height goes out of bounds
            if overlap_height + new_height > frame.shape[0]:
                new_height = frame.shape[0] - overlap_height

            # Update the total stitched image height dynamically
            stitched_height += new_height

            # Create a new canvas for the stitched image with updated size
            new_stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

            # Ensure that the slice for the current stitched image is valid and fits within the new canvas
            if self.stitched_image.shape[0] <= new_stitched_image.shape[0]:
                new_stitched_image[:self.stitched_image.shape[0], :stitched_width] = self.stitched_image
            else:
                print(f"Warning: current stitched image exceeds new canvas size at index {i}.")

            # Ensure that the slice for the frame is within valid bounds
            if overlap_height + new_height <= frame.shape[0]:
                new_stitched_image[self.stitched_image.shape[0]:self.stitched_image.shape[0] + new_height, :stitched_width] = frame[overlap_height:overlap_height + new_height, :stitched_width]

            # Update the stitched image with the new canvas
            self.stitched_image = new_stitched_image

        print("Stitching complete.")
        # Save the stitched image to the specified output path
        cv2.imwrite("LeafSegments/stitched_leaf.png", self.stitched_image)

# Usage example
video_path = "TestImages/leaf_output.mp4"  # Input video path
stitcher = LeafStitcher()
stitcher.process_video(video_path)

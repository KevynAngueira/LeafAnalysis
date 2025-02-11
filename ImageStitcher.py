import cv2
import numpy as np

class ImageStitcher:
    def __init__(self, contour_threshold=10, band_height=20, blend_height=10):
        self.contour_threshold = contour_threshold
        self.band_height = band_height
        self.blend_height = blend_height

        self.stitched_image = None
        self.prev_stitch_y= 0
    
    def crop_leaf(self, frame):
        """
        Crop out the black borders around the leaf using thresholding and contour detection.
        Assumes that the leaf is the largest bright region in the frame.
        Returns the cropped leaf image and its bounding box.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.contour_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, (0, 0, frame.shape[1], frame.shape[0])
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return frame[y:y+h, x:x+w], (x, y, w, h)

    def pad_image_to_match(self, image, target_shape):
        """
        Pads an image with zeroes to match the target shape.
        Returns the padded image and the y offset used in padding.
        """
        h, w = image.shape
        target_h, target_w = target_shape
        
        # Ensure the padded image is at least as high as band_height
        new_h = max(self.band_height, h)
        # Ensure the padded image is at least as wide as target_w
        new_w = max(target_w, w)
        
        # Expand canvas to new dimensions
        padded = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # Calculate padding dimensions and image location
        pad_h = (new_h - h)
        pad_w = (new_w - w)
        padding = (pad_h, pad_w)

        start_y = 0
        end_y = h

        start_x = 0
        end_x = w

        # Add image to padded canvas
        padded[start_y:end_y, start_x:end_x] = image
        
        return padded, padding

    def extract_bottom_band(self, image):
        """
        Extract a horizontal band (the lowest band) from the image.
        """
        return image[-self.band_height:, :]

    def blend_regions(self, stitched, target, y_start, blend_height):
        """ Feather blending over the transition region """
        alpha = np.linspace(1, 0, blend_height).reshape(-1, 1)
        for i in range(blend_height):
            stitched[y_start + i] = (alpha[i] * stitched[y_start + i] + (1 - alpha[i]) * target[i]).astype(np.uint8)

    def stitch_images(self, target_image):
        """ 
        Stitch images vertically using template matching to running image stitch
        
        self.stitched_image: the top segment (previous stitch)
        target_image: the bottom segment (new image to stich)
        """

        if self.stitched_image is None:
            self.stitched_image = target_image
            return self.stitched_image

        template_image = self.stitched_image[self.prev_stitch_y:, :]
        
        # Crop the leaf regions and get bounding boxes
        cropped_template, bbox_template = self.crop_leaf(template_image)
        cropped_target, bbox_target = self.crop_leaf(target_image)
        
        # Convert to grayscale for template matching
        gray_template = cv2.cvtColor(cropped_template, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(cropped_target, cv2.COLOR_BGR2GRAY)
        
        # Pad gray2 to match gray1's shape
        padded_target, padding = self.pad_image_to_match(gray_target, gray_template.shape)
        
        # Extract the bottom band of gray1 as the template
        template_band = self.extract_bottom_band(gray_template)
        
        # Template matching
        result = cv2.matchTemplate(padded_target, template_band, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"Template matching score: {max_val:.2f}") 

        if max_val > 0.2:

            # Determine cropped bottom of template
            template_bottom = self.prev_stitch_y + bbox_template[3] # height(stitched_image-prev_stitch) + height(cropped_template)
            print(f"Image1 Bottom: {template_bottom}")

            # Determine overlap height
            overlap_height = max_loc[1] # Stitch location in target
            print(f"Overlap: {overlap_height}")
            print(f'max_loc: {max_loc[0]}')

            # Determine stitch start y
            stitch_y = template_bottom - overlap_height
            print(f"Stitch Y: {stitch_y}")

            # Determine new dimensions
            overlap_band = min(overlap_height, self.band_height) # Removing the template band
            new_height = stitch_y + bbox_target[3] - overlap_band # stitch_y + height(cropped_target) - overlap_band
            stitched_width = max(self.stitched_image.shape[1], target_image.shape[1])

            # Create expanded stitch canvas
            new_stitched_image = np.zeros((new_height, stitched_width, 3), dtype=np.uint8)
            print(f"image1 shape: {self.stitched_image.shape}")
            print(f"stitched_image shape: {new_stitched_image.shape}")

            # Determine previous stitch coordinates in stitch canvas
            template_stitch_start_y, template_stitch_end_y = 0, template_bottom
            template_image_start_y, template_image_end_y = 0, template_bottom

            template_stitch_start_x, template_stitch_end_x = 0, self.stitched_image.shape[1]
            template_image_start_x, template_image_end_x = 0, self.stitched_image.shape[1]

            # Determine target_image coordinates in stitch canvas
            target_stitch_start_y, target_stitch_end_y = stitch_y, new_height
            target_image_start_y, target_image_end_y = bbox_target[1] + overlap_band, bbox_target[1]+bbox_target[3]

            target_stitch_start_x, target_stitch_end_x = 0, target_image.shape[1]
            target_image_start_x, target_image_end_x = 0, target_image.shape[1]
            
            # Stitch previous stitch and target_image together
            new_stitched_image[template_stitch_start_y:template_stitch_end_y, template_stitch_start_x:template_stitch_end_x] = self.stitched_image[template_image_start_y:template_image_end_y, template_image_start_x:template_image_end_x]
            new_stitched_image[target_stitch_start_y:target_stitch_end_y, target_stitch_start_x:target_stitch_end_x] = target_image[target_image_start_y:target_image_end_y, target_image_start_x:target_image_end_x]

            #self.blend_regions(new_stitched_image, target_image, stitch_y, self.blend_height)

            cv2.imshow("Stitched Video", new_stitched_image)
            cv2.imwrite("LeafSegments/stitched_leaf_CCORR.png", new_stitched_image)
        
            self.prev_stitch_y = stitch_y
            self.stitched_image = new_stitched_image
        
        #cv2.imshow("cropped_template", cropped_template)
        #cv2.moveWindow("cropped_template", 1000, 100)

        #cv2.imshow("Template Band", template_band)
        #cv2.moveWindow("cropped_template", 100, 700)

        #cv2.imshow("cropped_target", cropped_target)
        #cv2.moveWindow("cropped_target", 2000, 100)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        

        return self.stitched_image

    def process_video(self, video_path, frame_interval=10, video_duration=10):
        """
        Process a video of duration video_duration seconds.
        Extract a frame every frame_interval frames, resize each to 650x100,
        and stitch them sequentially.
        
        Returns the final stitched image.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(min(total_frames, fps * video_duration))
        
        stitched_result = None
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame_interval-th frame
            if frame_count % frame_interval == 0:
                # Resize frame to 650x100 if not already that size
                frame_resized = cv2.resize(frame, (650, 100))
                print(f"==== Processing frame {frame_count} ====")
                if stitched_result is None:
                    stitched_result = frame_resized
                else:
                    # Stitch the current stitched result with the new frame
                    stitched_result = self.stitch_images(frame_resized)
            
            frame_count += 1
        
        cap.release()
        return stitched_result

# Example usage:
video_path = "TestImages/leaf_output.mp4"  # Replace with your video file path
imageStitcher = ImageStitcher(contour_threshold=10, band_height=30, blend_height=10)
final_stitched_image = imageStitcher.process_video(video_path, frame_interval=30, video_duration=10)

if final_stitched_image is not None:
    cv2.imshow("Stitched Video Result", final_stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("LeafSegments/stitched_leaf_CCORR.png", final_stitched_image)
else:
    print("No frames were processed.")

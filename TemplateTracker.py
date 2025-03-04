import os
import shutil
import cv2
import numpy as np


class Template:
    def __init__(self, band, bbox):
        self.band = band
        self.bbox = bbox

class TemplateTracker:
    def __init__(self, move_threshold, output_path, frame_select=1, template_refresh=10, contour_threshold=10, band_height=20):
        self.move_threshold = move_threshold
        self.output_path = output_path

        self.frame_select = frame_select
        self.template_refresh = template_refresh

        self.contour_threshold = contour_threshold
        self.band_height = band_height

        self.prev_frame = None
        self.template = None
        self.move_tracker = 0
        self.saved_frames = 0
        self.frame_count = 0
        
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
            return gray, (0, 0, frame.shape[1], frame.shape[0]), False
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return gray[y:y+h, x:x+w], (x, y, w, h), True

    def extract_template_band(self, template):
        h, w = template.shape
        if h > self.band_height:
            start_y = (h//2) - (self.band_height // 2)
            end_y = (h//2) + (self.band_height // 2)
            template_band = template[start_y:end_y, :]
            return template_band, start_y
        else:
            return template, 0

    def detect_template(self, frame):
        cropped_leaf, cropped_bbox, leaf_found = self.crop_leaf(frame)
        #if leaf_found:
        template_band, band_start_y = self.extract_template_band(cropped_leaf)
        
        template_bbox = list(cropped_bbox)
        template_bbox[1] = cropped_bbox[1] + band_start_y
        template_bbox[3] = self.band_height

        cv2.imshow('Cropped', cropped_leaf)
        cv2.moveWindow('Cropped', 1000, 0)
        cv2.imshow('Template', template_band)
        cv2.moveWindow('Template', 1000, 500)

        self.template = Template(template_band, template_bbox)
        #else:
        #    self.template = None

    def pad_frame(self, frame, target_w):
        h, w = frame.shape
        new_h = max(self.band_height, h)
        new_w = max(target_w, w)
        padded = np.zeros((new_h, new_w), dtype=np.uint8)
        padded[:h, :w] = frame
        return padded

    def track_displacement(self, frame):
        if self.template is None:
            self.detect_template(frame)
            self.prev_frame = frame
            return False

        test_frame = frame.copy()
        
        # Check if the frame is completely black, and ignore it
        if np.all(frame == 0):
            print("⚠️ Black frame detected! Ignoring and not saving.")
            return False  

        print(f"== Processing Frame {self.frame_count} ==")
        if self.frame_count % self.frame_select == 0:
            template_w = self.template.bbox[2]

            cropped_frame, cropped_bbox, leaf_found = self.crop_leaf(frame)
            #if leaf_found:
            padded_frame = self.pad_frame(cropped_frame, template_w)

            # Template matching
            result = cv2.matchTemplate(padded_frame, self.template.band, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            displacement = abs(cropped_bbox[1] + max_loc[1] - self.template.bbox[1])
            self.move_tracker += displacement

            print(f"📏 Displacement: {displacement:.2f} pixels")
            print(f"Total Displacement: {self.move_tracker:.2f} pixels")
            print(f"Next Threshold: {self.move_threshold * self.saved_frames} pixels")

            if self.move_tracker >= (self.move_threshold * self.saved_frames):
                print(f"🆕 New Leaf Segment Detected! Saving frame {self.saved_frames}...")
                frame_path = os.path.join(self.output_path, f"leaf_segment_{self.saved_frames}.png")
                cv2.imwrite(frame_path, frame)
                self.saved_frames += 1
            
            cv2.line(test_frame, (0, int(max_loc[1])), (frame.shape[1], int(max_loc[1])), (255, 0, 0), 2)  # Blue line
            cv2.line(test_frame, (0, int(self.template.bbox[1])), (frame.shape[1], int(self.template.bbox[1])), (0, 0, 255), 2)  # Red line
            cv2.imshow('Leaf Tracking', test_frame)
            cv2.waitKey(1)  # Add this line to update the display continuously

            self.template.bbox[1] = cropped_bbox[1] + max_loc[1]            

        self.frame_count += 1
        if self.frame_count % self.template_refresh == 0:
            print("🔄 Refreshing template before computing displacement.")
            self.detect_template(frame)

        self.prev_frame = frame

def process_video(video_path, output_path="LeafSegments/"):
    """
    Processes a video frame by frame to detect and extract leaf area.
    If output_path is provided, saves the processed frames as a video.
    """

    tempTracker = TemplateTracker(100, output_path, frame_select = 1, template_refresh=1)

    cap = cv2.VideoCapture(video_path)

    # Debug: Check if video opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Stop when the video ends
        if not ret:
            break  

        tempTracker.track_displacement(frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    video_folder = 'demonstration5'
    video_path = f"TestVideos/{video_folder}/leaf_output.mp4"
    #video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join("LeafSegments/", video_folder)

    # Remove the folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True) 

    process_video(video_path, output_path)

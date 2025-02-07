import cv2
import numpy as np
from CropAndRotate import cropAndRotate
from DetectRectangle import detectRectangle
from ExtractGreen import extractGreen
from Helper.ResizeForDisplay import resize_for_display


class VideoDetection:
    def __init__(self, tool_hsv_thresholds, leaf_hsv_thresholds, target_dimensions, 
        move_threshold=0, large_move_threshold=50, confirm_frames=10, lost_frames=5,
    ):
        self.tool_hsv_thresholds = tool_hsv_thresholds
        self.leaf_hsv_thresholds = leaf_hsv_thresholds
        self.target_dimensions = target_dimensions

        self.target_rect = None
        self.target_box = None

        self.move_threshold = move_threshold 
        self.large_move_threshold = large_move_threshold
        self.confirm_frames = confirm_frames 
        self.lost_frames = lost_frames

        self.pending_target_box = None
        self.pending_target_rect = None
        self.frames_since_large_shift = 0 
        self.frames_since_last_seen = 0

    def update_target(self, new_target_box, new_target_rect):
        """ Update target based on changes """
        if new_target_rect is not None:
            if self.target_rect is None:
                # First detection
                self.target_box = new_target_box
                self.target_rect = new_target_rect
                self.frames_since_last_seen = 0
                self.frames_since_large_shift = 0
            else:
                # Calculate movement distance
                center_old = np.mean(self.target_box, axis=0)
                center_new = np.mean(new_target_box, axis=0)
                distance = np.linalg.norm(center_new - center_old)

                if distance < self.move_threshold:
                    # Small movement → Keep old detection
                    self.frames_since_last_seen = 0
                    self.frames_since_large_shift = 0  # Reset large shift counter
                elif distance < self.large_move_threshold:
                    # Medium movement → Accept immediately
                    self.target_box = new_target_box
                    self.target_rect = new_target_rect
                    self.frames_since_last_seen = 0
                    self.frames_since_large_shift = 0
                else:
                    # Large movement → Wait a few frames before updating
                    if self.frames_since_large_shift == 0:
                        self.pending_target_box = new_target_box
                        self.pending_target_rect = new_target_rect

                    self.frames_since_large_shift += 1

                    if self.frames_since_large_shift >= self.confirm_frames:
                        # Confirmed stable large move → Accept new detection
                        self.target_box = self.pending_target_box
                        self.target_rect = self.pending_target_rect
                        self.frames_since_large_shift = 0
                        self.frames_since_last_seen = 0
        else:
            self.frames_since_last_seen += 1

            if self.frames_since_last_seen > self.lost_frames:
                # Lost target for too long → Reset detection
                self.target_box = None
                self.target_rect = None
                self.frames_since_large_shift = 0

    def detect_frame(self, frame):
        """
        Processes a single frame to detect the rectangle and extract the leaf area.
        Returns the processed frame and the detected leaf area size.
        """

        frame_copy = frame.copy()
        new_target_box, new_target_rect = detectRectangle(frame, self.tool_hsv_thresholds, self.target_dimensions, draw=False)

        self.update_target(new_target_box, new_target_rect)

        # Process if a valid rectangle is found
        if self.target_rect is not None:
            cv2.drawContours(frame, [self.target_box], 0, (0, 255, 0), 3) 
            cv2.imshow("Target Contour", resize_for_display(frame))

            cropped_frame = cropAndRotate(frame_copy, self.target_rect)
            cv2.imshow("Cropped Frame", resize_for_display(cropped_frame))

            result, leaf_area = extractGreen(cropped_frame, self.tool_hsv_thresholds, self.leaf_hsv_thresholds, self.target_dimensions)

            return result, leaf_area
        else:
            return None, 0  # No detection


    def process_video(self, video_path, output_path=None):
        """
        Processes a video frame by frame to detect and extract leaf area.
        If output_path is provided, saves the processed frames as a video.
        """

        cap = cv2.VideoCapture(video_path)

        # Debug: Check if video opened
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        frame_width = 650
        frame_height = 100
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the video writer if saving output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            
            # Stop when the video ends
            if not ret:
                break  

            result, leaf_area = self.detect_frame(frame)
            print(f"Leaf Area: {leaf_area} pixels")

            if result is not None:
                resized_result = cv2.resize(result, (frame_width, frame_height))
                display_result = resize_for_display(resized_result)
                cv2.imshow("Processed Frame", display_result)

                # Write frame to output video if saving
                if output_path:
                    out.write(resized_result)

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "TestImages/demonstration.mp4"
    #output_path = "TestImages/cropped_output.mp4"
    output_path = None

    # Define tool and leaf adaptive hsv mask
    orange_hsv_thresholds = (np.array([2, 115, -1]), np.array([12, 255, 255]))
    leaf_hsv_thresholds = (np.array([35, 23, 0]), np.array([100, 255, 215]))
    # Define target area dimensions
    target_dimensions = (6.5, 1)

    VideoDetection = VideoDetection(orange_hsv_thresholds, leaf_hsv_thresholds, target_dimensions)
    VideoDetection.process_video(video_path, output_path)



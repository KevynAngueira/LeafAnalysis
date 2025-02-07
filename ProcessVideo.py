import cv2
import numpy as np
from CropAndRotate import cropAndRotate
from DetectRectangle import detectRectangle
from ExtractGreen import extractGreen
from Helper.ResizeForDisplay import resize_for_display


class VideoDetection:
    def __init__(self, tool_hsv_thresholds, leaf_hsv_thresholds, target_dimensions):
        self.target_rect = None
        self.target_box = None

        self.tool_hsv_thresholds = tool_hsv_thresholds
        self.leaf_hsv_thresholds = leaf_hsv_thresholds
        self.target_dimensions = target_dimensions

    def detect_frame(self, frame):
        """
        Processes a single frame to detect the rectangle and extract the leaf area.
        Returns the processed frame and the detected leaf area size.
        """

        frame_copy = frame.copy()

        if self.target_box is None:
            # Detect the target rectangle
            target_box, target_rect = detectRectangle(frame, self.tool_hsv_thresholds, self.target_dimensions, draw=False)

        # Process if a valid rectangle is found
        if target_rect is not None:
            cv2.drawContours(frame, [target_box], 0, (0, 255, 0), 3) 
            cv2.imshow("Target Contour", resize_for_display(frame))

            cropped_frame = cropAndRotate(frame_copy, target_rect)
            cv2.imshow("Cropped Frame", resize_for_display(cropped_frame))

            result, leaf_area = extractGreen(cropped_frame, orange_hsv_thresholds, leaf_hsv_thresholds, rectangle_dimensions)

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
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
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
                resized_result = resize_for_display(result)
                cv2.imshow("Processed Frame", resized_result)
                #cv2.waitKey(0)

                # Write frame to output video if saving
                if output_path:
                    out.write(result)

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "TestImages/demonstration.mp4"
    output_path = None

    # Define tool and leaf adaptive hsv mask
    orange_hsv_thresholds = (np.array([2, 115, -1]), np.array([12, 255, 255]))
    leaf_hsv_thresholds = (np.array([35, 23, 0]), np.array([100, 255, 215]))
    # Define target area dimensions
    target_dimensions = (6.5, 1)

    VideoDetection = VideoDetection(orange_hsv_thresholds, leaf_hsv_thresholds, target_dimensions)
    VideoDetection.process_video(video_path, output_path)



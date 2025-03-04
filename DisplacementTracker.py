import cv2
import numpy as np

class DisplacementTracker:
    def __init__(self, move_threshold, frame_select=5, frame_refresh=10):
        self.move_threshold = move_threshold
        self.frame_select = frame_select
        self.frame_refresh = frame_refresh

        self.prev_y_positions = None # Store previous y positions of features
        self.prev_gray = None  # Store previous grayscale frame
        self.features = None    # Store detected features
        self.saved_frames = 0   # Count saved segments
        self.frame_count = 0    # Count frames before next refresh 
        self.skipped_frames = 0  # Count the number of frames skipped
        self.move_tracker = 0   # The displacement of features

    def detect_features(self, frame):
        """ Detect strong features on the leaf using Shi-Tomasi corner detection. """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.4, minDistance=7)
        if self.features is not None:
            self.prev_y_positions = self.features[:, 0, 1]  # Store initial Y-coordinates
        self.prev_gray = gray  # Store first frame

    def track_features(self, frame, enter_threshold=25, exit_threshold=25):
        """ Track features using Optical Flow and detect when a new leaf segment is visible. """
        if self.prev_gray is None or self.features is None:
            print("Extracting Features!")
            self.detect_features(frame)

            return False

         # Check if the frame is completely black, and ignore it
        if np.all(frame == 0):
            print("⚠️ Black frame detected! Ignoring and not saving.")
            return False  

        print(f"== Processing Frame {self.frame_count} ==")
        if self.frame_count % self.frame_select == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.features, None)
            
            if new_features is not None:
                # Get indices of valid features from optical flow
                valid_idx = np.where(status.flatten() == 1)[0]
                valid_features = new_features[valid_idx].reshape(-1, 2)
                valid_prev_y = self.prev_y_positions[valid_idx]  # Filter baseline using the same indices

                height, width, _ = frame.shape

                # Apply threshold filtering to both valid_features and valid_prev_y
                mask = (valid_features[:, 1] <= height - enter_threshold) & (valid_features[:, 1] >= exit_threshold)
                valid_features = valid_features[mask]
                valid_prev_y = valid_prev_y[mask]

                # Edge Case: if no valid features remain, re-detect features and reset move tracker
                if len(valid_features) == 0:
                    self.detect_features(frame)
                    self.skipped_frames += 1
                    self.move_tracker += self.move_tracker / self.frame_count
                    print(f"Avg Movement: {self.move_tracker / self.frame_count}")
                    print(f"No Valid Features! {self.skipped_frames}")
                    self.frame_count += 1
                    return False

                # Calculate cumulative vertical displacement using the filtered features
                displacement = np.abs(valid_features[:, 1] - valid_prev_y)
                avg_movement = np.mean(displacement)

                print(f"📏 Average Feature Movement: {avg_movement:.2f} pixels")
                self.move_tracker += avg_movement
                print(f'Move Tracker: {self.move_tracker}')
                print(f'Move Threshold: {self.move_threshold}')

                # New Segment Case: Displacement matches segment height, new segement detected
                if self.move_tracker >= (self.move_threshold * self.saved_frames):
                    print(f"🆕 New Leaf Segment Detected! Saving frame {self.saved_frames}...")
                    cv2.imwrite(f"LeafSegments/leaf_segment_{self.saved_frames}.png", frame)
                    self.saved_frames += 1
                    # Reset tracking
                    self.detect_features(frame)
                
                # No Segment Case: Displacement not reached, keep tracking
                else: 
                    self.features = valid_features.reshape(-1, 1, 2) 
                    self.prev_y_positions = self.features[:, 0, 1]

                # Visualization: Draw the features
                for feature in valid_features:
                    cv2.circle(frame, tuple(feature.astype(int)), 5, (0, 255, 0), -1)  # Green dots for valid features

                # Draw the exit threshold line
                cv2.line(frame, (0, int(height - enter_threshold)), (frame.shape[1], int(height - enter_threshold)), (255, 0, 0), 2)  # Blue line
                cv2.line(frame, (0, int(exit_threshold)), (frame.shape[1], int(exit_threshold)), (0, 0, 255), 2)  # Red line

            self.prev_gray = gray
            
            # Display the frame with features and threshold line
            cv2.imshow('Leaf Tracking', frame)
            cv2.waitKey(1)  # Add this line to update the display continuously

        # Increase frame counter and refresh features BEFORE computing displacement
        self.frame_count += 1
        if self.frame_count % self.frame_refresh == 0:
            print("🔄 Refreshing features before computing displacement.")
            self.detect_features(frame)

def process_video(video_path, output_path=None):
        """
        Processes a video frame by frame to detect and extract leaf area.
        If output_path is provided, saves the processed frames as a video.
        """
        # 1 5
        # 1 10
        distTracker = DisplacementTracker(100, frame_select = 1, frame_refresh=15)

        cap = cv2.VideoCapture(video_path)

        # Debug: Check if video opened
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
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

            distTracker.track_features(frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "TestImages/leaf_output.mp4"
    
    process_video(video_path)

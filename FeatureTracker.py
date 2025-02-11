import cv2
import numpy as np

class FeatureTracker:
    def __init__(self):
        self.prev_gray = None  # Store previous grayscale frame
        self.features = None    # Store detected features
        self.saved_frames = 0   # Count saved segments

    def detect_features(self, frame):
        """ Detect strong features on the leaf using Shi-Tomasi corner detection. """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        self.prev_gray = gray  # Store first frame

    def track_features(self, frame, enter_threshold=20, exit_threshold=20):
        """ Track features using Optical Flow and detect when a new leaf segment is visible. """
        if self.prev_gray is None or self.features is None:
            print("Extracting Features!")
            self.detect_features(frame)
            return False


         # Check if the frame is completely black, and ignore it
        if np.all(frame == 0):
            print("⚠️ Black frame detected! Ignoring and not saving.")
            return False  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.features, None)
        
        if new_features is not None:
            valid_features = new_features[status.flatten() == 1]
            valid_features = valid_features.reshape(-1, 2)

            height, width, _ = frame.shape
            # ❌ Ignore above and below the eixt and entrance thresholds respectively
            valid_features = valid_features[valid_features[:, 1] <= height - enter_threshold]
            valid_features = valid_features[valid_features[:, 1] >= exit_threshold]
            
            # 🔍 If no valid features remain, assume a new segment is visible
            if len(valid_features) == 0:
                print(f"🆕 New Leaf Segment Detected! Checking if frame should be saved...")
                
                # ✅ Only save if new features are detected after re-extracting
                self.detect_features(frame)
                if self.features is not None and len(self.features) > 0:
                    print(f"✅ Saving frame {self.saved_frames}.")
                    cv2.imwrite(f"LeafSegments/leaf_segment_{self.saved_frames}.png", frame)
                    self.saved_frames += 1
            else:
                self.features = valid_features.reshape(-1, 1, 2)  # Keep tracking

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

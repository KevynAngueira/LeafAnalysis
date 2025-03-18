# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-03-18


import cv2
import numpy as np

from Scripts.LeafSeparator import LeafSeparator, LeafSeparatorConfig
from Scripts.StabilizedLeafSeparator import StabilizedLeafSeparator

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig
from Scripts.StabilizedViewWindow import StabilizedViewWindow


from Scripts.ResizeForDisplay import resize_for_display

class LeafScan:
    def __init__(self, leaf_config: LeafSeparatorConfig = None, view_config: ViewWindowConfig = None):
        if leaf_config is None:
            leaf_config = LeafSeparatorConfig()
        if view_config is None:
            view_config = ViewWindowConfig()

        self.target_dimensions = leaf_config.target_dimensions
        
        #self.leafSeparator = LeafSeparator(leaf_config)
        self.leafSeparator = StabilizedLeafSeparator(leaf_config)

        #self.viewWindow = ViewWindow(view_config)
        self.viewWindow = StabilizedViewWindow(view_config)

    def processFrame(self, frame, frame_count, output_path):
        
        if frame_count > 700 and (frame_count % 20) == 0:
            view_window = self.viewWindow.Extract(frame, True)
            leaf_result, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)
        else:
            view_window = self.viewWindow.Extract(frame)
            leaf_result, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)
        
        cv2.imshow("Frame", resize_for_display(frame))
        cv2.imshow("View Window", resize_for_display(view_window))
        cv2.imshow("Leaf Result", resize_for_display(leaf_result))

        # Write frame to output video if saving
        #if output_path and (frame_count%20)==0:
        #    cv2.imwrite(f"{output_path}/view_{frame_count}.jpg", view_window)


    def scanVideo(self, video_path, output_path=None):
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
        frame_width = self.target_dimensions[0]
        frame_height = self.target_dimensions[1]
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

            self.processFrame(frame, frame_count, output_path)

            # Write frame to output video if saving
            #if output_path and (frame_count%20)==0:
                #out.write(frame)
            #    cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)

            frame_count += 1

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

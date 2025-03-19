# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-03-19


import cv2
import numpy as np

from Scripts.LeafSeparator import LeafSeparator, LeafSeparatorConfig
from Scripts.StabilizedLeafSeparator import StabilizedLeafSeparator

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig
from Scripts.StabilizedViewWindow import StabilizedViewWindow

from Scripts.SegmentDetector import SegmentDetector

from Scripts.ResizeForDisplay import resize_for_display

class LeafScan:
    def __init__(self, view_config: ViewWindowConfig = None, leaf_config: LeafSeparatorConfig = None, output_folder=None):
        
        if view_config is None:
            view_config = ViewWindowConfig()
        if leaf_config is None:
            leaf_config = LeafSeparatorConfig()

        #self.viewWindow = ViewWindow(view_config)
        self.viewWindow = StabilizedViewWindow(view_config)
        
        #self.leafSeparator = LeafSeparator(leaf_config)
        self.leafSeparator = StabilizedLeafSeparator(leaf_config)
        self.target_dimensions = leaf_config.target_dimensions

        self.segmentDetector = SegmentDetector(output_folder)

    def processFrame(self, frame, frame_count, output_path):
        
        if frame_count > 700 and (frame_count % 20) == 0:
            view_window = self.viewWindow.Extract(frame, True)
            leaf_result, leaf_mask, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)
        else:
            view_window = self.viewWindow.Extract(frame)
            leaf_result, leaf_mask, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)
            
            #band, band_mask = self.segmentDetector.extractTemplate(leaf_result, leaf_mask)
            #drawn_template, max_loc = self.segmentDetector.templateMatching(leaf_result, band, band_mask)

            drawn_template = self.segmentDetector.detectSegment(leaf_result, leaf_mask)
        
        cv2.imshow("Frame", resize_for_display(frame))
        cv2.imshow("View Window", resize_for_display(view_window))
        cv2.imshow("Leaf Result", resize_for_display(leaf_result))
        #cv2.imshow("Band", resize_for_display(band))
        #cv2.imshow("Band Mask", resize_for_display(band_mask))
        cv2.imshow("Drawn Template", resize_for_display(drawn_template))

        return leaf_result

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

            result = self.processFrame(frame, frame_count, output_path)

            # Write frame to output video if saving
            #if output_path:
            #    out.write(result)
            #    cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)

            frame_count += 1

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

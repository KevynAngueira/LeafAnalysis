# Author: Kevyn Angueira Irizarry
# Created: 2025-03-17
# Last Modified: 2025-05-01

import os
import cv2
import numpy as np

from Scripts.LeafSeparator import LeafSeparator, LeafSeparatorConfig
from Scripts.StabilizedLeafSeparator import StabilizedLeafSeparator

from Scripts.ViewWindow import ViewWindow, ViewWindowConfig
from Scripts.StabilizedViewWindow import StabilizedViewWindow

from Scripts.SegmentDetector import SegmentDetector
from Scripts.PhaseCorrelationDetector import PhaseCorrelationDetector

from Scripts.LeafAreaCalculator import LeafAreaCalculator
from Scripts.LeafWidthExtractor import LeafWidthExtractor

from Scripts.ResizeForDisplay import resize_for_display

class LeafScan:
    def __init__(self, view_config: ViewWindowConfig = None, leaf_config: LeafSeparatorConfig = None, output_folder=None, display=False):
        
        if view_config is None:
            view_config = ViewWindowConfig()
        if leaf_config is None:
            leaf_config = LeafSeparatorConfig()
        self.display = display

        #self.viewWindow = ViewWindow(view_config)
        self.viewWindow = StabilizedViewWindow(view_config)
        
        #self.leafSeparator = LeafSeparator(leaf_config)
        self.leafSeparator = StabilizedLeafSeparator(leaf_config)
        self.target_dimensions = leaf_config.target_dimensions

        self.output_folder = output_folder

        self.segmentDetector = SegmentDetector()
        #self.segmentDetector = PhaseCorrelationDetector(output_folder)

        window_dimensions = (self.target_dimensions[0] / 100.0, self.target_dimensions[1] / 100.0)
        self.leafAreaCalculator = LeafAreaCalculator(window_dimensions)
        self.leafWidthExtractor = LeafWidthExtractor()

    def processFrame(self, frame, output_path, out=None):
        
        if self.frame_count >= 2000 and self.frame_count % 1 == 0:
            view_window = self.viewWindow.Extract(frame, display=True)
            leaf_result, leaf_mask, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)
        else:
            view_window = self.viewWindow.Extract(frame)
            leaf_result, leaf_mask, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window)

        _, drawn_template = self.segmentDetector.trackCummulativeDisplacement(leaf_result, leaf_mask)

        #if is_new_segment:
        #    leaf_area = self.leafAreaCalculator.calculateSegment(leaf_mask)
        #    self.leafWidthExtractor.extractWidth(leaf_mask)
        #    print(f"Area: {leaf_area}")

        #print(self.frame_count)

        if self.display:
            cv2.imshow("Frame", resize_for_display(frame))
            cv2.imshow("View Window", resize_for_display(view_window))
            cv2.imshow("Leaf Result", resize_for_display(leaf_mask))
            cv2.imshow("Drawn Template", resize_for_display(drawn_template))

        #if output_path and frame_count % 5 == 0:
        #    cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", view_window)

        if out is not None:
            out.write(leaf_result)

        self.frame_count += 1

        return leaf_result

    def reset(self, full_reset=False):
        self.leafSeparator.resetLeafSeparator()
        self.viewWindow.resetViewWindow()

        if full_reset:
            self.leafAreaCalculator.resetAreas()
            self.segmentDetector.resetSegments()
            self.leafWidthExtractor.resetWidths()
            self.frame_count = 0

    def processVideo(self, video_path, output_path=None):
        """
        Processes a leaf video frame by frame:
        (1) Crops video to the target view window
        (2) Separates leaf pixels from background
        (3) Tracks vertical displacement for new leaf segment starts
        """

        self.reset(True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Set video properties
        frame_width, frame_height = self.target_dimensions
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # If saving output, define video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()

            # Stop when the video ends
            if not ret:
                break

            result = self.processFrame(frame, output_path, out)
            
            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def extractResults(self,remaining_leaf_length, video_path):
        """
        Save the unique segments, calculate remaining area, and extract the base widths
        """
        self.reset(False)

        # Get the indexes of the unique leaf segments
        segment_indexes = self.segmentDetector.getSegmentIndexes(remaining_leaf_length)

        # Setup second video trace 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Extract the unique leaf segments, calculate their area and widths, and save them
        for segment_idx, frame_idx in enumerate(segment_indexes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Fast jump to frame
            ret, frame = cap.read()

            if ret:
                view_window = self.viewWindow.Extract(frame, display=False, stabilize=False)
                leaf_result, leaf_mask, leaf_pixels, leaf_percentage = self.leafSeparator.Extract(view_window, display=False, stabilize=False)

                leaf_area = self.leafAreaCalculator.calculateSegment(leaf_mask)
                self.leafWidthExtractor.extractWidth(leaf_mask)

                print(f"Unique Segment Detected! | Frame Index = {frame_idx} AND Segment Area = {leaf_area}")
                
                if self.output_folder is not None:
                    output_path = os.path.join(self.output_folder, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(output_path, leaf_result)
            else:
                print(f"Warning: Failed to grab frame at index {frame_idx}")

        cap.release()
        print("Done.")

        print()

        #base_widths = self.leafWidthExtractor.getWidths()
        #print(f"Base Widths: {base_widths}")

        print()

        all_areas = self.leafAreaCalculator.getAllAreas()
        print("All Areas:")
        print(all_areas)

        print()

        total_area = self.leafAreaCalculator.getTotalArea()
        print(f"Total Area: {total_area}")

        return total_area #, base_widths
    
    def scanVideo(self, remaining_leaf_length, video_path, output_path=None):
        self.processVideo(video_path, output_path)
        total_area = self.extractResults(remaining_leaf_length, video_path)

        return total_area #, base_widths

# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-09-24

import os
import cv2
import numpy as np

from LeafScan.Utils import resize_for_display

from LeafScan.Configs import ViewExtractorConfig, LeafExtractorConfig

from LeafScan.Core.ViewExtractor import ViewExtractor, StabilizedViewExtractor
from LeafScan.Core.LeafExtractor import LeafExtractor, KmeansLeafExtractor, StabilizedLeafExtractor
from LeafScan.Core.SliceDetector import FOpticalFlowDetector as SliceDetector
from LeafScan.Core.SliceAreaCalculator import SliceAreaCalculator

class LeafScan:
    def __init__(self, view_config: ViewExtractorConfig = None, leaf_config: LeafExtractorConfig = None, output_folder=None, display=False, deep_display=False):
        
        if view_config is None:
            view_config = ViewExtractorConfig()
        if leaf_config is None:
            leaf_config = LeafExtractorConfig()
        self.display = display
        self.deep_display = deep_display
        self.output_folder = output_folder

        self.target_dimensions = leaf_config.target_dimensions

        self.viewExtractor = StabilizedViewExtractor(view_config)
        self.leafExtractor = StabilizedLeafExtractor(leaf_config)
        self.sliceDetector = SliceDetector()
        self.areaCalculator = SliceAreaCalculator()

    def processFrame(self, frame, output_path, out=None):
        
        view_window = self.viewExtractor.Extract(frame, display=False)
        leaf_result, leaf_mask = self.leafExtractor.Extract(view_window, display=self.deep_display)
        total_displacement, vis_displacement = self.sliceDetector.trackCummulativeDisplacement(leaf_result, leaf_mask)
        
        if self.display:
            #print(self.frame_count)
            #print("Total_Displacement: ", total_displacement)
            cv2.imshow("Frame", resize_for_display(frame))
            
            cv2.imshow("View Window", view_window)
            cv2.imshow("Visualized Displacement", resize_for_display(vis_displacement))

        if out is not None:
            if view_window.dtype != np.uint8:
                view_window = (drawn_template * 255).clip(0, 255).astype(np.uint8)
            if len(view_window.shape) == 2:
                view_window = cv2.cvtColor(view_window, cv2.COLOR_GRAY2BGR)
            out.write(view_window)

        self.frame_count += 1

        return leaf_result, leaf_mask

    def reset(self, full_reset=False):
        self.frame_count = 0
        self.viewExtractor.reset()
        
        if full_reset:
            self.sliceDetector.reset()
            self.areaCalculator.reset()

    def processVideo(self, video_path, output_path=None):
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
        Save the unique slices and calculate full leaf area
        """
        self.reset(False)

        # Get the indexes of the unique leaf slices
        slice_indexes = self.sliceDetector.getSliceIndexes(remaining_leaf_length)

        # Setup second video trace 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Extract the unique leaf slices, calculate their area and widths, and save them
        for slice_idx, frame_idx in enumerate(slice_indexes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Fast jump to frame
            ret, frame = cap.read()

            if ret:
                view_window = self.viewExtractor.Extract(frame, display=False, stabilize=False)
                leaf_result, leaf_mask = self.leafExtractor.Extract(view_window, display=False, stabilize=False)

                #cv2.imshow(f"View Windoq {slice_idx}", resize_for_display(view_window))
                #cv2.imshow(f"Slice Index {slice_idx}", resize_for_display(leaf_result))
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                leaf_area = self.areaCalculator.calculateArea(leaf_mask)
                
                print(f"Unique slice Detected! | Frame Index = {frame_idx} AND slice Area = {leaf_area}")
                
                if self.output_folder is not None:
                    output_path = os.path.join(self.output_folder, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(output_path, leaf_result)
            else:
                print(f"Warning: Failed to grab frame at index {frame_idx}")

        cap.release()

        all_areas = self.areaCalculator.getAllAreas()
        total_area = self.areaCalculator.getTotalArea()

        print(f'All Areas:\n{all_areas}')
        print(f'Total Area:\n{total_area}')

        return total_area
    
    def scanVideo(self, remaining_leaf_length, video_path, output_path=None):
        self.processVideo(video_path, output_path)
        total_area = self.extractResults(remaining_leaf_length, video_path)

        return total_area
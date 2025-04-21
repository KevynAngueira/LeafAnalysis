# Author: Kevyn Angueira Irizarry
# Created: 2025-04-07
# Last Modified: 2025-04-21



import pandas as pd
from dataclasses import dataclass, field
from typing import List

@dataclass
class LeafDataConfig:
    leaf_file: str = 'LeafAreas.xlsx'
    measurements_sheet: str = 'LeafMeasurements'
    areas_sheet: str = 'LeafAreas'
    measurements_headers: List[str] = field(default_factory=lambda: ['LeafID', 'SegmentID', 'Start_Width', 'End_Width', 'Area'])
    areas_headers: List[str] = field(default_factory=lambda: ['LeafID', 'Area'])

class LeafData:
    def __init__(self, config: LeafDataConfig=None):
        if config is None:
            config = LeafDataConfig()

        self.__dict__.update(vars(config))

    def getLeafByID(self, leaf_id):
        # Read the XLSX file
        df = pd.read_excel(self.leaf_file, sheet_name=self.measurements_sheet, engine='openpyxl')

        # Filter out any rows where LeafID is NaN or empty
        df = df[df[self.measurements_headers[0]].notna()]

        # Filter the rows that match the given LeafID
        leaf_data = df[df['LeafID'] == leaf_id]

        # Clean the data by removing unnecessary columns and rows with missing values in critical columns
        leaf_data = leaf_data[self.measurements_headers]

        # Check if we have data for the given LeafID
        if leaf_data.empty:
            raise ValueError(f"Error: No leaf data found for LeafID {leaf_id}. Check the XLSX file.")

        return leaf_data

    def getWidthsByID(self, leaf_id):
        leaf_data = self.getLeafByID(leaf_id)

        # Get all start_widths and the last end_width
        widths = leaf_data[self.measurements_headers[2]].tolist() 
        last_end_width = float(leaf_data[self.measurements_headers[3]].iloc[-1])
        widths.append(last_end_width)

        return widths
    
    def getAreaByID(self, leaf_id):
        # Read the XLSX file
        df = pd.read_excel(self.leaf_file, sheet_name=self.areas_sheet, engine='openpyxl')

        # Filter out any rows where LeafID is NaN or empty
        df = df[df[self.areas_headers[0]].notna()]

        # Filter the rows that match the given LeafID
        leaf_data = df[df['LeafID'] == leaf_id]

        # Clean the data by removing unnecessary columns and rows with missing values in critical columns
        leaf_data = leaf_data[self.areas_headers]

        # Check if we have data for the given LeafID
        if leaf_data.empty:
            raise ValueError(f"Error: No leaf data found for LeafID {leaf_id}. Check the XLSX file.")
        
        # Return the area as a float
        return float(leaf_data['Area'].iloc[0])
    
    def getLengthByID(self, leaf_id):
        leaf_data = self.getLeafByID(leaf_id)

        start_widths = list(leaf_data["Start_Width"])
        end_widths = list(leaf_data["End_Width"])
        segment_count = len(start_widths)

        # Default length is the number of segments (1 inch per segment)
        if segment_count < 2:
            effective_length = segment_count
        else:
            last_start = start_widths[-1]
            second_last_end = end_widths[-2]

            # If the last segment is tapering, adjust its length
            if last_start < second_last_end and second_last_end > 0:
                ratio = last_start / second_last_end
                effective_length = (segment_count - 1) + ratio
            else:
                effective_length = segment_count

        return effective_length

if __name__ == "__main__":
    leafData = LeafData()
    
    leaf_data = leafData.getLeafByID(1)
    print(leaf_data)

    leaf_widths = leafData.getWidthsByID(1)
    print(leaf_widths)

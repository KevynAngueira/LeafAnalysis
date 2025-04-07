import os
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from shapely.geometry import Polygon, LineString

from LeafData import LeafDataConfig
from LeafPlotter import LeafPlotter

from dataclasses import dataclass

@dataclass
class DefoliationModellerConfig:
    grid_size: float = 0.25
    no_defo_zone: float = 2
    cut_tolerance: float = 0.5
    slice_tolerance: float = 0.5
    retries: int = 30

class DefoliationModeller(LeafPlotter):
    def __init__(self, data_config: LeafDataConfig = None, defo_config: DefoliationModellerConfig = None):
        super().__init__(data_config)

        if defo_config is None:
            defo_config = DefoliationModellerConfig()

        self.__dict__.update(vars(defo_config))

    def createGrid(self, outline):
        """
        Creates a grid of potential cut positions, spaced by `grid_size` inches.
        Ensures no cuts are placed in the first `no_defo_zone` inches along the y-axis.
        """
        x_outline, y_outline = outline
        
        # Adjust x_min and x_max to the nearest 0.25 increments
        x_min = np.floor(min(x_outline) * 4) / 4  
        x_max = np.ceil(max(x_outline) * 4) / 4 

        y_min, y_max = min(y_outline), max(y_outline)

        print(x_min, x_max)
        print(y_min, y_max)

        grid_positions = []
        for x in np.arange(x_min, x_max, self.grid_size):
            for y in np.arange(y_min + self.no_defo_zone, y_max, self.grid_size):  # Skip no_defo_zone
                grid_positions.append((x, y))
        return grid_positions

    def applyCutsFromOutline(self, outline, cut_percentage):
        """
        Apply cuts (rectangular areas) to the leaf shape to remove a percentage of the leaf's total area.
        """
        x_outline, y_outline = outline

        # Step 1: Calculate the total area of the leaf
        leaf_shape = Polygon(zip(x_outline, y_outline))
        total_area = abs(leaf_shape.area)
        print(total_area)

        # Step 2: Calculate the target area to remove
        target_area_to_remove = total_area * cut_percentage / 100
        area_tolerance = total_area * self.cut_tolerance / 100

        # Step 3: Define possible cut dimensions (in inches) and grid
        cut_dimensions = [0.25, 0.5, 0.75, 1.0]
        grid_positions = self.createGrid(outline)

        # Step 4: Randomly select cuts and apply them
        cut_area_removed = 0
        placed_cuts = []

        retry_counter = 0
        while (retry_counter < self.retries) and (abs(target_area_to_remove - cut_area_removed) > area_tolerance):
            retry_counter += 1

            # Randomly pick a grid position
            x, y = random.choice(grid_positions)

            # Randomly select a cut dimensions
            rect_height = random.choice(cut_dimensions)
            rect_width = random.choice(cut_dimensions)

            # Define the selected cut
            rect = Polygon([(x, y), (x + rect_width, y), (x + rect_width, y + rect_height), (x, y + rect_height)])
            intersection = rect.intersection(leaf_shape)

            # Check Leaf Overlap: Check if the cut overlaps with the leaf
            if not intersection.is_empty:
                
                # Check Cut Overlap: Check if the cut does not overlap with other cuts
                if all(not rect.intersects(cut) for cut in placed_cuts):
                    overlap_area = intersection.area

                    # Check Large Cut: Check if the cut is not too large
                    if (target_area_to_remove + area_tolerance) > (cut_area_removed + overlap_area):

                        # Add cut to list and add leaf overlap to removed area
                        placed_cuts.append(rect)
                        cut_area_removed += overlap_area
                        print(f"Cut added. Overlap area: {overlap_area}. Total Cut Area: {cut_area_removed}")  
                        
                        retry_counter = 0

                        # Remove cut position from the possible positions
                        grid_positions.remove((x,y))

        print(f"Intended Removal: {target_area_to_remove}")
        print(f"Total Area Removed: {cut_area_removed}")

        return placed_cuts

    def applyCutsFromID(self, leaf_id, cut_percentage):
        """
        Apply cuts (rectangular areas) to the leaf shape to remove a percentage of the leaf's total area.
        """

        outline = self.leafShapeFromID(leaf_id)
        placed_cuts = self.applyCutsFromOutline(outline, cut_percentage)

        return placed_cuts

    def applySlicesFromOutline(self, outline, slice_percentage):
        """
        Apply slicing transformations to the leaf shape based on the perimeter increase percentage.
        """

        x_outline, y_outline = outline
        
        # Step 1: Calculate the total perimeter of the leaf
        leaf_shape = Polygon(zip(x_outline, y_outline))
        initial_perimeter = leaf_shape.length
        print(initial_perimeter)

        # Step 2: Calculate the target perimeter to increase
        target_perimeter_increase = initial_perimeter * slice_percentage / 100
        perimeter_tolerance = initial_perimeter * self.slice_tolerance / 100

         # Step 3: Define possible slice heights (in inches) and positions
        max_y = max(y_outline)

        x_min = np.floor(min(x_outline) * 4) / 4  
        x_max = np.ceil(max(x_outline) * 4) / 4 
        
        slice_positions = list(np.arange(x_min, x_max, 0.25))
        slice_heights = list(np.arange(max_y, self.no_defo_zone, -1))

        # Step 4: Randomly select slices and apply them
        applied_slices = []

        retry_counter = 0
        perimeter_increase = 0
        
        while (retry_counter < self.retries) and (abs(target_perimeter_increase - perimeter_increase) > perimeter_tolerance):
            retry_counter += 1

            # Randomly pick a grid position
            x = random.choice(slice_positions)

            # Randomly select a slice height
            s_height = random.choice(slice_heights)
            min_y = s_height

            slice_line = LineString([(float(x), float(min_y)), (float(x), float(max_y))])

            intersection = slice_line.intersection(leaf_shape)
            
            if not intersection.is_empty:
                overlap_perimeter = 2*intersection.length

                # Check Large Cut: Check if the cut is not too large
                if (target_perimeter_increase + perimeter_tolerance) > (perimeter_increase + overlap_perimeter):
                   
                    applied_slices.append(slice_line)
                    perimeter_increase += overlap_perimeter
                    print(f"Cut added. Overlap Perimeter: {overlap_perimeter}. Total Sliced Perimeter: {perimeter_increase}")  

                    slice_positions.remove(x)
                    retry_counter = 0

        print(f"Intended Slice Increase: {target_perimeter_increase}")
        print(f"Total Slice Increase: {perimeter_increase}")

        return applied_slices

    def applySlicesByID(self, leaf_id, slice_percentage):
        """
        Apply slicing transformations based on leaf ID.
        """

        outline = self.leafShapeFromID(leaf_id)
        applied_slices = self.applySlicesFromOutline(outline, slice_percentage)

        return applied_slices

    def PlotCut(self, leaf_id, cut_percentage):
        """
        Plots the generated leaf shape based on input widths. 
        Additionally apply placed cuts
        """ 

        outline = self.leafShapeFromID(leaf_id)
        x_outline, y_outline = outline

        placed_cuts = self.applyCutsFromOutline(outline, cut_percentage)

        plt.plot(x_outline, y_outline, 'g-', linewidth=2, label="Original Leaf Shape")
        
        # Draw the cuts
        for cut in placed_cuts:
            x, y = cut.exterior.xy
            plt.fill(x, y, color='red', alpha=0.7)

        plt.fill(x_outline, y_outline, color='green', alpha=0.5, label="Leaf Shape After Cuts")
        plt.ylabel("Length (inches)")
        plt.xlabel("Width (inches)")
        plt.title("Leaf Shape with Random Cuts Applied")
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(f"leaf{leaf_id:03}_cut_plot.png")  # Saves the plot to a PNG file

        return placed_cuts  # Return the cuts made for reference

    def PlotSlices(self, leaf_id, slice_percentage):
        """
        Plots the generated leaf shape based on input widths. 
        Additionally applies placed slices.
        """ 

        outline = self.leafShapeFromID(leaf_id)
        x_outline, y_outline = outline

        placed_slices = self.applySlicesFromOutline(outline, slice_percentage)

        plt.plot(x_outline, y_outline, 'g-', linewidth=2, label="Original Leaf Shape")

        # Draw the slices
        for line_slice in placed_slices:
            x, y = line_slice.xy 
            plt.plot(x, y, color='red', alpha=0.7, linewidth=2) 

        plt.fill(x_outline, y_outline, color='green', alpha=0.5, label="Leaf Shape After Slices")
        plt.ylabel("Length (inches)")
        plt.xlabel("Width (inches)")
        plt.title("Leaf Shape with Slices Applied")
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(f"leaf{leaf_id:03}_slice_plot.png")  # Saves the plot to a PNG file

        return placed_slices  # Return the slices made for reference        

if  __name__ == "__main__":
    defoModeller = DefoliationModeller()
    
    # placed_cuts = defoModeller.PlotCut(1, 20)
    
    placed_slices = defoModeller.PlotSlices(1, 60)
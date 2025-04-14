# Author: Kevyn Angueira Irizarry
# Created: 2025-04-07
# Last Modified: 2025-04-14

import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import trapezoid
from shapely.geometry import Polygon, LineString, MultiPolygon, box

from LeafData import LeafDataConfig
from LeafPlotter import LeafPlotter

from dataclasses import dataclass
from typing import List

@dataclass
class DefoliationModellerConfig:
    slice_cell_size: float = 0.25
    grid_cell_size: float = 0.25
    no_defo_zone: int = 3

    possible_cell_selections: int = 6

    slice_tolerance: float = 0.02
    cut_tolerance: float = 0.02
    brown_tolerance: float = 0.02
    retries: int = 30

class DefoliationModeller(LeafPlotter):
    """
    Applies defoliation transformations (slicing, cutting, browning) to a given leaf
    """
    def __init__(self, data_config: LeafDataConfig = None, defo_config: DefoliationModellerConfig = None):
        super().__init__(data_config)

        if defo_config is None:
            defo_config = DefoliationModellerConfig()

        self.__dict__.update(vars(defo_config))

        self.slice_positions = []
        self.grid_positions = []

        # Slicing: Increases shape complexity, w/o reducing area
        self.placed_slices = []
        self.target_perimeter_increase = 0
        self.perimeter_increase = 0
        self.perimeter_tolerance = 0

        self.slice_y_max = 0
        self.possible_slice_heights = []
        
        # Cutting: Reduces area
        self.placed_cuts = []
        self.target_removed_area = 0
        self.removed_area = 0
        self.removed_area_tolerance = 0

        self.possible_cut_dimensions = []
    
        # Browining: Creates dead tissue areas
        self.placed_brownings = []
        self.target_browned_area = 0
        self.browned_area = 0
        self.browned_area_tolerance = 0

        self.possible_brown_dimensions = []

        # Transformation Tracker: Backtracking
        self.transformation_tracker = []

        self.leaf_id = None
        self.leaf_outline = None
        self.leaf_shape = None
        self.transformed_leaf_shape = None
        self.retry_counter = 0

    def gridPositionsFromOutline(self, outline):
        """
        Creates a grid of potential cut and browinin positions, spaced by `grid_cell_size` inches.
        Ensures no cuts are placed in the first `no_defo_zone` inches along the y-axis.
        """
        x_outline, y_outline = outline
        
        # Define grid position ranges
        x_min= np.floor(min(x_outline) / self.grid_cell_size) * self.grid_cell_size  
        x_max = np.ceil(max(x_outline) / self.grid_cell_size) * self.grid_cell_size 

        y_min = np.floor(min(y_outline) / self.grid_cell_size) * self.grid_cell_size  
        y_max = np.ceil(max(y_outline) / self.grid_cell_size) * self.grid_cell_size 

        # Define possible grid possitions
        grid_positions = [(x, y) for x in np.arange(x_min, x_max, self.grid_cell_size)  
                        for y in np.arange(y_min + self.no_defo_zone, y_max, self.grid_cell_size)]

        return grid_positions

    def slicePositionsFromOutline(self, outline):
        """
        Creates an array of potential slice positions, spaced by 'slice_cell_size' inches.
        """
        x_outline, _ = outline

        # Define slice position range
        x_min = np.floor(min(x_outline) / self.slice_cell_size) * self.slice_cell_size  
        x_max = np.ceil(max(x_outline) / self.slice_cell_size) * self.slice_cell_size 

        # Define possible slice positions
        slice_positions = list(np.arange(x_min, x_max, self.slice_cell_size))

        return slice_positions

    def _maxPerimeterIncrease(self):
        """
        Calculates the max possible perimeter increase from slicing on a given leaf
        """
        max_perimeter_increase = 0
        for position in self.slice_positions:
            vertical_line = LineString([
                (position, -1e6),
                (position, 1e6)
            ])
            overlap = vertical_line.intersection(self.leaf_shape)
            max_perimeter_increase += overlap.length
        return max_perimeter_increase

    def initSlices(self, slice_percentage):
        """
        Prepares all values necessary for slicing transformations
        """

        # Setup possible slice positions
        self.slice_positions = self.slicePositionsFromOutline(self.leaf_outline)
        
        # Setup target and tolerance thresholds
        max_perimeter_increase = self._maxPerimeterIncrease()
        self.target_perimeter_increase = max_perimeter_increase * slice_percentage / 100
        self.perimeter_tolerance = self.target_perimeter_increase * self.slice_tolerance
        self.perimeter_increase = 0

        # Setup possible slice heights and slice start y
        self.slice_y_max = max(self.leaf_outline[1])
        self.possible_slice_heights = list(np.arange(self.no_defo_zone, self.slice_y_max, 1))


    def initCuts(self, cut_percentage):
        """
        Prepares all values necessary for cut transformations.
        (Except the grid which is shared by cuts and brownings)
        """

        # Setup targer and tolerance thresholds
        initial_area = abs(self.leaf_shape.area)
        self.target_removed_area = initial_area * cut_percentage / 100
        self.removed_area_tolerance = self.target_removed_area * self.cut_tolerance
        self.removed_area = 0

        # Setup possible dimensions choices
        cut_max = self.possible_cell_selections * self.grid_cell_size
        self.possible_cut_dimensions = list(np.arange(2*self.grid_cell_size, cut_max, self.grid_cell_size))

    def initBrownings(self, brown_percentage):
        """
        Prepares all values necessary for browning transformations.
        (Except the grid which is shared by cuts and brownings)
        """
        brown_percentage = brown_percentage if brown_percentage > 0 else 0

        # Setup targer and tolerance thresholds
        initial_area = abs(self.leaf_shape.area)
        self.target_browned_area = initial_area * brown_percentage / 100
        self.browned_area_tolerance = self.target_browned_area * self.brown_tolerance
        self.browned_area = 0

        # Setup possible dimensions choices
        brown_max = self.possible_cell_selections * self.grid_cell_size
        self.possible_brown_dimensions = list(np.arange(2*self.grid_cell_size, brown_max, self.grid_cell_size))


    def initLeaf(self, leaf_id, slice_percentage, cut_percentage, brown_percentage):
        """
        Prepares all values outline, slice_positions, and grid_positions to apply defoliation trasnformations
        on the new leaf.
        """

        # Get new leaf outline/shape
        leaf_outline = self.leafShapeFromID(leaf_id)
        self.leaf_outline = leaf_outline
        self.leaf_id = leaf_id
        self.transformation_tracker = []
        
        x_outline, y_outline = self.leaf_outline
        self.leaf_shape = Polygon(zip(x_outline, y_outline))
        self.transformed_leaf_shape = Polygon(zip(x_outline, y_outline))

        # Setup Slices
        self.initSlices(slice_percentage)

        # Setup Grid
        self.grid_positions = self.gridPositionsFromOutline(leaf_outline)
        
        # Setup Cuts
        self.initCuts(cut_percentage)

        # Setup Brownings
        self.initBrownings(brown_percentage)

    def attemptTrans(self, trans_geom):
        """
        Attempt to apply transformation (cut or slice) to self.transformed_leaf_shape

        Only applies if transformation would preserve shape contiguity, rejects otherwise
        """
         # Buffer LineStrings to make them subtractable
        if isinstance(trans_geom, LineString):
            trans_geom = trans_geom.buffer(0.001)

        # Apply the transformation to a copy
        temp_shape = self.transformed_leaf_shape.difference(trans_geom)

        # Check if the new shape is a valid, contiguous Polygon
        if isinstance(temp_shape, Polygon) and temp_shape.is_valid:
            self.transformed_leaf_shape = temp_shape
            return True
        else:
            return False

    def applySlice(self):
        """
        Applies a singular slice to the leaf.
        """

        # Check if perimeter is within tolerance
        remaining_increase = self.target_perimeter_increase - self.perimeter_increase
 
        if remaining_increase < -self.perimeter_tolerance:
            # Case Tolerance Exceeded: Remove slice
            self.removeSlice()

            # Ensure to remove the last slice from the transformation tracker
            if self.transformation_tracker and "s" in self.transformation_tracker:
                for i in range(len(self.transformation_tracker)-1):
                    if self.transformation_tracker[i] == "s":
                        self.transformation_tracker.pop(i)
                        break

            return False
        

        if remaining_increase > self.perimeter_tolerance:
            # Case Perimeter Not Reached: Add slice
                
            slice_was_placed = False

            # Keep trying to place a new slice until max retries are reached
            while not slice_was_placed and (self.retry_counter < self.retries):

                self.retry_counter += 1

                x = random.choice(self.slice_positions)
                slice_height = random.choice(self.possible_slice_heights)

                y_max = max(self.leaf_outline[1])
                y_min = y_max - slice_height

                new_slice_line = LineString([(float(x), float(y_min)), (float(x), float(y_max))])
                
                # Check if slice line naively intersects with the leaf shape
                intersection = new_slice_line.intersection(self.leaf_shape) 
                if not intersection.is_empty and intersection.length > 0:
                    # Case Intersects Leaf: continue

                    # Invalidate length portions that intersect with existing cuts
                    naive_leaf_overlap = intersection
                    for cut in self.placed_cuts:
                        naive_leaf_overlap = naive_leaf_overlap.difference(cut)

                    # Check if slice line actually intersects with the leaf shape
                    if not naive_leaf_overlap.is_empty and naive_leaf_overlap.length > 0:
                        # Case Intersects Leaf: continue    
                        overlap_perimeter = 2 * naive_leaf_overlap.length

                        # Check if slice line's added perimeter does surpass the tolerance threshold
                        is_under_tolerance = (self.perimeter_increase + overlap_perimeter) <= (self.target_perimeter_increase + self.perimeter_tolerance)
                        if is_under_tolerance:
                            # Case Under Tolerance: Continue

                            # Check if slice preserves shape contiguity
                            if self.attemptTrans(new_slice_line):
                                # Case Preserves Contiguity: Place Line

                                self.placed_slices.append(new_slice_line)
                                self.perimeter_increase += overlap_perimeter
                                self.slice_positions.remove(x)

                                self.transformation_tracker.append("s")
                                self.retry_counter = 0
                                slice_was_placed = True

                                print(f"Slice added. Overlap Area: {overlap_perimeter}.")
                                print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")
                                print(x, slice_height)

            return False
        return True

    def applyCut(self):
        """
        Applies a singular cut to the leaf.
        """
        
        # Check if area is within tolerance
        if abs(self.target_removed_area - self.removed_area) > self.removed_area_tolerance:
            # Case Tolerance Not Reached: Keep cutting

            cut_was_placed = False

            # Keep trying to place a new cut until max retries are reached
            while not cut_was_placed and (self.retry_counter < self.retries):
                
                self.retry_counter += 1

                x, y = random.choice(self.grid_positions)
                cut_height = random.choice(self.possible_cut_dimensions)
                cut_width = random.choice(self.possible_cut_dimensions)

                new_cut = Polygon([(x, y), (x + cut_width, y), (x + cut_width, y + cut_height), (x, y + cut_height)])
                
                # Check if cut intersects with the leaf shape
                intersection = new_cut.intersection(self.leaf_shape)
                if not intersection.is_empty:
                    # Case Intersects Leaf: continue

                    # Check if cut intersects with any of the placed slices, cuts, or browning
                    if (
                        all(not new_cut.intersection(cut).area > 0 for cut in self.placed_cuts) and
                        all(not new_cut.intersection(browning).area > 0 for browning in self.placed_brownings)
                    ): 
                        # Case No Intersections: continue
                        
                        overlap_area = intersection.area

                        # Check if cut's added area exceeds the tolerance threshold
                        is_under_tolerance = (self.removed_area + overlap_area) <= (self.target_removed_area + self.removed_area_tolerance)
                        if is_under_tolerance:
                            # Case Under Tolerance: Place Cut

                            # Check if cut preserves shape contiguity
                            if self.attemptTrans(new_cut):
                                # Case Preserves Contiguity

                                self.placed_cuts.append(new_cut)
                                self.removed_area += overlap_area
                                self.grid_positions.remove((x,y))

                                # Remove perimeter invalidated by the cut
                                for slice_line in self.placed_slices:
                                    naive_slice_overlap = slice_line.intersection(self.leaf_shape)
                                    slice_overlap = naive_slice_overlap.intersection(new_cut)
                                    invalidated_perimeter = 2 * slice_overlap.length 
                                    self.perimeter_increase -= invalidated_perimeter

                                self.transformation_tracker.append("c")
                                self.retry_counter = 0
                                cut_was_placed = True

                                print(f"Cut added. Overlap Area: {overlap_area}.")
                                print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")
                                print(x, y, cut_height, cut_width)

            return False
        return True


    def applyBrowning(self):
        """
        Applies a singular browning to the leaf
        """

        # Check if area is within tolerance
        if abs(self.target_browned_area - self.browned_area) > self.browned_area_tolerance:
            # Case Tolerance Not Reached: Keep cutting

            browning_was_placed = False

            # Keep trying to place a new browining until max retries are reached
            while not browning_was_placed and (self.retry_counter < self.retries):
                
                self.retry_counter += 1

                x, y = random.choice(self.grid_positions)
                brown_height = random.choice(self.possible_brown_dimensions)
                brown_width = random.choice(self.possible_brown_dimensions)

                new_brown = Polygon([(x, y), (x + brown_width, y), (x + brown_width, y + brown_height), (x, y + brown_height)])
                
                # Check if browning intersects with the leaf shape
                intersection = new_brown.intersection(self.leaf_shape)
                if not intersection.is_empty:
                    # Case Intersects Leaf: continue

                    # Check if cut intersects with any of the placed slices, cuts, or browning
                    if (
                        all(not new_brown.intersection(cut).area > 0 for cut in self.placed_cuts) and
                        all(not new_brown.intersection(browning).area > 0 for browning in self.placed_brownings)
                    ): 
                        # Case No Intersections: continue
                        
                        overlap_area = intersection.area

                        # Check if cut's added area exceeds the tolerance threshold
                        is_under_tolerance = (self.browned_area + overlap_area) <= (self.target_browned_area + self.browned_area_tolerance)
                        if is_under_tolerance:
                            # Case Under Tolerance: Place Browning

                            self.placed_brownings.append(new_brown)
                            self.browned_area += overlap_area
                            self.grid_positions.remove((x,y))

                            self.transformation_tracker.append("b")
                            self.retry_counter = 0
                            browning_was_placed = True

                            print(f"Browning added. Overlap Area: {overlap_area}.")
                            print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")
                            print(x, y, brown_height, brown_width)
            return False
        return True

    def revertTrans(self, trans_geom):
        """
        Reverts applied transformation (cut or slice) on self.transformed_leaf_shape
        """
        if isinstance(trans_geom, LineString):
            trans_geom = trans_geom.buffer(0.001)

        self.transformed_leaf_shape = self.transformed_leaf_shape.union(trans_geom)

    def removeSlice(self):
        """
        Removes the last slice that was applied.
        """
        if not self.placed_slices:
            return 

        #last_slice_line = self.placed_slices.pop()
        i = random.randrange(len(self.placed_slices))
        last_slice_line = self.placed_slices.pop(i)

        # Remove the portion of the slice line not invalidated by cuts
        naive_leaf_overlap = last_slice_line.intersection(self.leaf_shape)
        for cut in self.placed_cuts:
            naive_leaf_overlap = naive_leaf_overlap.difference(cut)
        overlap_perimeter = 2 * naive_leaf_overlap.length

        self.perimeter_increase -= overlap_perimeter

        x = last_slice_line.coords[0][0]
        self.slice_positions.append(x)

        self.revertTrans(last_slice_line)

        self.retry_counter = 0

        print(f">>>> REMOVED SLICE <<<<")
        print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")

    def removeCut(self):
        """
        Removes the last cut that was applied.
        """
        if not self.placed_cuts:
            return 

        #last_cut = self.placed_cuts.pop()
        i = random.randrange(len(self.placed_cuts))
        last_cut = self.placed_cuts.pop(i)

        #Re-add the portion of the slices no longer invalidated by the cut
        for slice_line in self.placed_slices:
            naive_slice_overlap = slice_line.intersection(self.leaf_shape)
            slice_overlap = naive_slice_overlap.intersection(last_cut)
            revalidated_perimeter = 2 * slice_overlap.length
            self.perimeter_increase += revalidated_perimeter

        overlap_area = last_cut.intersection(self.leaf_shape).area
        self.removed_area -= overlap_area

        x, y = last_cut.exterior.coords[0]
        self.grid_positions.append((x,y))

        self.revertTrans(last_cut)

        self.retry_counter = 0
        
        print(f">>>> REMOVED CUT <<<<")
        print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")

    def removeBrowning(self):
        """
        Removes the last brownin that was applied.
        """
        if not self.placed_brownings:
            return 

        # last_browning = self.placed_brownings.pop(0)
        i = random.randrange(len(self.placed_brownings))
        last_browning = self.placed_brownings.pop(i)

        overlap_area = last_browning.intersection(self.leaf_shape).area
        self.browned_area -= overlap_area

        x, y = last_browning.exterior.coords[0]
        self.grid_positions.append((x,y))

        self.retry_counter = 0

        print(f">>>> REMOVED BROWNING <<<<")
        print(f"[{self.perimeter_increase}, {self.removed_area}, {self.browned_area}]")

    def sanityCheck(self, slice_percentage, cut_percentage, brown_percentage):
        total_area = self.leaf_shape.area

        min_x, min_y, max_x, max_y = self.leaf_shape.bounds
        no_defo_rect = Polygon([(min_x, min_y), (max_x, min_y), (max_x, self.no_defo_zone), (min_x, self.no_defo_zone)])
    
        transformation_space = self.leaf_shape.difference(no_defo_rect)
        max_percentage = transformation_space.area / total_area * 100

        combined_percentage = slice_percentage + cut_percentage + brown_percentage

        is_possible = max_percentage >= combined_percentage
        
        #return is_possible, combined_percentage, max_percentage
        return True, combined_percentage, max_percentage

    def show_interactive_browning_selector(self, grid_positions, grid_cell_size):
        fig, ax = plt.subplots()

        # Plot the leaf shape (assumed to be a shapely Polygon)
        x, y = self.leaf_shape.exterior.xy
        ax.plot(x, y, color='green', linewidth=2)
        ax.set_aspect('equal')

        selected_cells = set()
        cell_patches = []

        # Create rectangle patches for each cell
        for idx, (x, y) in enumerate(grid_positions):
            rect_patch = patches.Rectangle((x, y), grid_cell_size, grid_cell_size,
                                        linewidth=1, edgecolor='gray', facecolor='none')
            ax.add_patch(rect_patch)
            cell_patches.append(rect_patch)

        def on_click(event):
            if event.inaxes != ax:
                return

            for idx, (x, y) in enumerate(grid_positions):
                if x <= event.xdata <= x + grid_cell_size and y <= event.ydata <= y + grid_cell_size:
                    if idx in selected_cells:
                        selected_cells.remove(idx)
                        cell_patches[idx].set_facecolor('none')
                    else:
                        selected_cells.add(idx)
                        cell_patches[idx].set_facecolor('orange')
                    fig.canvas.draw_idle()
                    break

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.title("Click grid cells to select browned regions")
        plt.show()

        return selected_cells

    def calculate_browning_percentage(self, selected_cells):
        browned_area = 0.0

        for idx in selected_cells:
            x, y = self.grid_positions[idx]
            cell_polygon = box(x, y, x + self.grid_cell_size, y + self.grid_cell_size)
            browned_area += cell_polygon.intersection(self.leaf_shape).area

        return browned_area / self.leaf_shape.area  # Returns the percentage

    def transformLeaf(self, leaf_id, slice_percentage, cut_percentage, brown_percentage):
        """
        Applies backtracking algorithm with predefined browned cells.
        Alternates between slicing and cutting only. Browned cells are fixed and unmodifiable.
        """

        is_done = []
        applyTransformations = []
        removeTransformations = {}

        transformation_map = {
            "s": (self.applySlice, self.removeSlice, slice_percentage),
            "c": (self.applyCut, self.removeCut, cut_percentage),
            "b": (self.applyBrowning, self.removeBrowning, brown_percentage),
        }

        for key, (apply_func, remove_func, percentage) in transformation_map.items():
            if percentage > 0:
                is_done.append(False)
                applyTransformations.append(apply_func)
                removeTransformations[key] = remove_func

        # Initialize leaf without a brown_percentage
        self.initLeaf(leaf_id, slice_percentage, cut_percentage, brown_percentage)

        if brown_percentage < 0:
            # Case Predetermined Browning: Enter predetermined brown cells            
            selected_cells = self.show_interactive_browning_selector(self.grid_positions, self.grid_cell_size)

            # Apply predetermined browning transformations
            for idx in selected_cells:
                x, y = self.grid_positions[idx]
                cell_poly = box(x, y, x + self.grid_cell_size, y + self.grid_cell_size)
                self.browned_area += cell_poly.intersection(self.leaf_shape).area
                self.placed_brownings.append(cell_poly)

            # Skip sanity check for brownings, instead calculate percentage from selected cells
            brown_percentage = self.calculate_browning_percentage(selected_cells)

        is_possible, combined_percentage, max_percentage = self.sanityCheck(slice_percentage, cut_percentage, brown_percentage)
        if not is_possible:
            print("Error: The combined Percentage exceeds the maximum possible in the leaf space")
            print(f"Combined Percentage: {combined_percentage} | Max Percentage: {max_percentage}")
            return 

        transformation_counter = 0
        reset_counter = 0

        while not all(is_done) and transformation_counter < 1500:
            print(self.retry_counter)
            t_index = transformation_counter % len(is_done)

            if self.retry_counter >= self.retries and self.transformation_tracker:
                t_to_remove = self.transformation_tracker.pop(0)
                removeTransformations[t_to_remove]()

            is_done[t_index] = applyTransformations[t_index]() 
            transformation_counter += 1

        print("==== DEFOLIATION TRACKER ====")
        print(self.transformation_tracker)

        print("==== SLICES ====")
        print(f"Target Perimeter Increase: {self.target_perimeter_increase}")
        print(f"Perimeter Increase: {self.perimeter_increase}")

        print("==== CUTS ====")
        print(f"Target Removed Area: {self.target_removed_area}")
        print(f"Removed Area: {self.removed_area}")

        print("==== BROWNINGS ====")
        print(f"Target Browned Area: {self.target_browned_area}")
        print(f"Browned Area: {self.browned_area}")

        print("==== TRANSFORMATION COUNTER ====")
        print(transformation_counter)

    def Plot(self):
        """
        Plots the applied transformations with a more visible and adjustable grid.
        Includes darker grid lines at every 1-inch interval.
        """

        outline = self.leaf_outline
        x_outline, y_outline = outline

        plt.figure(figsize=(8, 8))

        # Draw the leaf first to keep it in the background
        plt.fill(x_outline, y_outline, color='green', alpha=0.5, label="Leaf")

        # Highlight the center column
        center_x = (min(x_outline) + max(x_outline)) / 2
        plt.axvline(x=center_x, color='blue', alpha=0.7, linewidth=2, label="Center Column")

        # Draw the slices
        slice_drawn, cut_drawn, browning_drawn = False, False, False

        for line_slice in self.placed_slices:
            x, y = line_slice.xy 
            plt.plot(x, y, color='red', alpha=0.7, linewidth=2, label="Slices" if not slice_drawn else "")
            slice_drawn = True

        # Draw the cuts
        for cut in self.placed_cuts:
            x, y = cut.exterior.xy
            plt.fill(x, y, color='black', alpha=0.7, label="Cuts" if not cut_drawn else "")
            cut_drawn = True

        # Draw the brownings
        for browning in self.placed_brownings:
            x, y = browning.exterior.xy
            plt.fill(x, y, color='saddlebrown', alpha=0.7, label="Brownings" if not browning_drawn else "")
            browning_drawn = True

        # Calculate grid ranges
        x_min = np.floor(min(x_outline))
        x_max = np.ceil(max(x_outline))
        y_min = np.floor(min(y_outline))
        y_max = np.ceil(max(y_outline))

        # Light grid lines at self.grid_cell_size
        x_ticks = np.arange(x_min, x_max + self.grid_cell_size, self.grid_cell_size)
        y_ticks = np.arange(y_min, y_max + self.grid_cell_size, self.grid_cell_size)
        plt.xticks(x_ticks, rotation=90)
        plt.yticks(y_ticks)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

        # Darker grid lines at every inch
        inch_x = np.arange(x_min, x_max + 1, 1)
        inch_y = np.arange(y_min, y_max + 1, 1)

        ax = plt.gca()
        for x in inch_x:
            ax.axvline(x=x, color='gray', linestyle='-', linewidth=1.2, alpha=0.9)
        for y in inch_y:
            ax.axhline(y=y, color='gray', linestyle='-', linewidth=1.2, alpha=0.9)

        # Aspect ratio and labels
        ax.set_aspect('equal', adjustable='datalim')
        plt.xlabel("Width (inches)")
        plt.ylabel("Length (inches)")
        plt.title("Leaf Shape with Transformations")
        plt.legend()

        # Save the figure
        plt.savefig(f"leaf_plots/leaf{self.leaf_id:03}_trans_plot.png")


if  __name__ == "__main__":
    if len(sys.argv) != 5:
        print("File usage is: python3 DefoliationModeller.py <leaf_id> <slice percentage>, <cut_percentage>, <browning_percentage>")
        quit()

    defoModeller = DefoliationModeller()
    
    placed_slices = defoModeller.transformLeaf(*map(int, sys.argv[1:]))
    defoModeller.Plot()

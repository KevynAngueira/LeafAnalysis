# Author: Kevyn Angueira Irizarry
# Created: 2025-04-07
# Last Modified: 2025-04-07

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from shapely.geometry import Polygon

import random

def read_leaf_data(file_path, leaf_id):
    # Read the XLSX file
    df = pd.read_excel(file_path, sheet_name='LeafMeasurements', engine='openpyxl')

    # Filter out any rows where LeafID is NaN or empty
    df = df[df['LeafID'].notna()]

    # Filter the rows that match the given LeafID
    leaf_data = df[df['LeafID'] == leaf_id]
    
    # Clean the data by removing unnecessary columns and rows with missing values in critical columns
    leaf_data = leaf_data[['LeafID', 'SegmentID', 'Start_Width', 'End_Width', 'Average_Width', 'Area']]

    # Check if we have data for the given LeafID
    if leaf_data.empty:
        raise ValueError(f"Error: No leaf data found for LeafID {leaf_id}. Check the XLSX file.")

    print(leaf_data)
    
    return leaf_data

def get_leaf_widths(leaf_data):
    # Check if leaf_data is empty or not
    if leaf_data.empty:
        raise ValueError("Error: No leaf widths found! Check the data.")
    
    # Get all start_widths and the last end_width
    widths = leaf_data['Start_Width'].tolist()  # Extract all Start_Width values
    last_end_width = leaf_data['End_Width'].iloc[-1]  # Get the last End_Width value
    widths.append(last_end_width)  # Add the last End_Width to the list

    return widths

def create_leaf_shape(widths):
    """
    Creates an approximate shape of a leaf given widths at each inch interval.
    """
    num_points = len(widths)
    y = np.arange(num_points)  # Each inch along the leaf's length (y-axis)
    x_upper = np.array(widths) / 2  # Upper boundary (half the width)
    x_lower = -x_upper  # Lower boundary (mirrored)
    
    # Create the outline by connecting upper and lower parts
    y_outline = np.concatenate([y, y[::-1]])  # Length is on the y-axis now
    x_outline = np.concatenate([x_upper, x_lower[::-1]])  # Width is on the x-axis now
    
    return x_outline, y_outline

def plot_leaf(widths, leaf_id):
    """
    Plots the generated leaf shape based on input widths.
    """
    x_outline, y_outline = create_leaf_shape(widths)
    
    plt.figure(figsize=(12, 6))  # Adjust the aspect ratio for a horizontal layout
    plt.plot(x_outline, y_outline, 'g-', linewidth=2)
    plt.fill(x_outline, y_outline, color='green', alpha=0.5)
    plt.ylabel("Length (inches)")  # Length on the y-axis
    plt.xlabel("Width (inches)")  # Width on the x-axis
    plt.title("Approximate Leaf Shape")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"leaf{leaf_id:03}_shape_plot.png")  # Saves the plot to a PNG file

    # Open the saved image and keep the window open until a key is pressed
    plt.show()

def create_cut_grid(x_range, y_range, grid_size=0.25, no_cut_zone=2):
    """
    Creates a grid of potential cut positions, spaced by `grid_size` inches.
    Ensures no cuts are placed in the first `no_cut_zone` inches along the y-axis.
    """
    grid_positions = []
    for x in np.arange(x_range[0], x_range[1], grid_size):
        for y in np.arange(y_range[0] + no_cut_zone, y_range[1], grid_size):  # Skip first 3 inches (no_cut_zone)
            grid_positions.append((x, y))
    return grid_positions

def apply_cut_to_leaf(x_outline, y_outline, cut_area_percentage, grid_size=0.25, tolerance=0.5, retry=30):
    """
    Apply cuts (rectangular areas) to the leaf shape to remove a percentage of the leaf's total area.
    """
    # Step 1: Calculate the total area of the leaf
    leaf_shape = Polygon(zip(x_outline, y_outline))
    total_area = abs(leaf_shape.area)  
    print(total_area)

    # Step 2: Calculate the target area to remove
    target_area_to_remove = total_area * cut_area_percentage / 100
    area_tolerance = total_area * tolerance / 100
    
    
    # Step 3: Define possible cut dimensions
    cut_dimensions = [0.25, 0.5, 0.75, 1.0]  # Possible cut dimensions in inches

    # Step 4: Randomly select cuts and apply them
    cut_area_removed = 0
    placed_cuts = []

    # Create a grid of potential cut positions within the leaf bounds
    x_min, x_max = min(x_outline), max(x_outline)
    y_min, y_max = min(y_outline), max(y_outline)
    
    grid_positions = create_cut_grid((x_min, x_max), (y_min, y_max), grid_size)

    # Keep adding cuts until we reach the target area to remove
    retry_counter = 0
    while retry_counter < retry and abs(target_area_to_remove - cut_area_removed) > area_tolerance:
        retry_counter += 1

        # Randomly pick a grid position
        x, y = random.choice(grid_positions)

        # Randomly select a cut dimension
        rect_height = random.choice(cut_dimensions)
        rect_width = random.choice(cut_dimensions)

        # The randomly selected cut
        rect = Polygon([(x, y), (x + rect_width, y), (x + rect_width, y + rect_height), (x, y + rect_height)])
        
        # Chect if the cut overlaps with the leaf
        intersection = rect.intersection(leaf_shape)
        if not intersection.is_empty:
            # Check if it overlaps with any previously placed cuts
            if all(not rect.intersects(cut) for cut in placed_cuts):
                overlap_area = intersection.area # Ovelap between cut and the leaf

                # Check cut removes too much area
                if (target_area_to_remove + area_tolerance) > (cut_area_removed + overlap_area):
                    # Place the cut rectangle and reduce the remaining area to be cut
                    placed_cuts.append(rect)
                    cut_area_removed += overlap_area

                    # Remove the cut from the grid (if you want to prevent reusing the same spot)
                    grid_positions.remove((x, y))

                    retry_counter = 0

                    # Print out the progress
                    print(f"Cut added. Overlap area: {overlap_area}. Total Cut Area: {cut_area_removed}")                   

    print(f"Total Area Removed: {cut_area_removed}")
    print(f"Intended Removal: {target_area_to_remove}")

    # Step 5: Create the plot of the leaf with the cuts
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
    plt.show()

    return placed_cuts  # Return the cuts made for reference

# Example usage:
file_path = "../LeafAreas.xlsx"
cut_area_percentage = 20
leaf_id = 3
leaf_data = read_leaf_data(file_path, leaf_id)
leaf_widths = get_leaf_widths(leaf_data)
print("Extracted widths:", leaf_widths)
#plot_leaf(leaf_widths, leaf_id)

x_outline, y_outline = create_leaf_shape(leaf_widths)

# Apply cuts to the leaf shape
apply_cut_to_leaf(x_outline, y_outline, cut_area_percentage)

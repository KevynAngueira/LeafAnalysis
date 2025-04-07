# Author: Kevyn Angueira Irizarry
# Created: 2025-04-07
# Last Modified: 2025-04-07

import numpy as np
import matplotlib.pyplot as plt

def create_leaf_shape(widths):
    """
    Creates an approximate shape of a leaf given widths at each inch interval.
    """
    num_points = len(widths)
    x = np.arange(num_points)  # Each inch along the leaf's length
    y_upper = np.array(widths) / 2  # Upper boundary (half the width)
    y_lower = -y_upper  # Lower boundary (mirrored)
    
    # Create the outline by connecting upper and lower parts
    x_outline = np.concatenate([x, x[::-1]])
    y_outline = np.concatenate([y_upper, y_lower[::-1]])
    
    return x_outline, y_outline

def plot_leaf(widths):
    """
    Plots the generated leaf shape based on input widths.
    """
    x_outline, y_outline = create_leaf_shape(widths)
    
    plt.figure(figsize=(6, 12))
    plt.plot(x_outline, y_outline, 'g-', linewidth=2)
    plt.fill(x_outline, y_outline, color='green', alpha=0.5)
    plt.xlabel("Length (inches)")
    plt.ylabel("Width (inches)")
    plt.title("Approximate Leaf Shape")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("leaf_shape_plot.png")  # Saves the plot to a PNG file

# Example usage:
leaf_widths = [1, 2, 2.5, 3, 3.2, 3, 2.8, 2, 1, 0.5]  # Example widths at each inch interval
plot_leaf(leaf_widths)

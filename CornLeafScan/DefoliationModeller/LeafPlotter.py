# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import os
import numpy as np
import matplotlib.pyplot as plt

from LeafData import LeafData, LeafDataConfig

class LeafPlotter:
    def __init__(self, data_config: LeafDataConfig = None, output_folder=None):
        self.leafData = LeafData(data_config)
        self.output_folder = output_folder

    def leafShapeFromWidths(self, widths):
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
    
    def leafShapeFromID(self, leaf_id):
        """
        Creates an approximate shape of a leaf given a leaf ID.
        """
        widths = self.leafData.getWidthsByID(leaf_id)
        return self.leafShapeFromWidths(widths)

    def Plot(self, leaf_id):
        """
        Plots the generated leaf shape based on input widths.
        """ 
        outline = self.leafShapeFromID(leaf_id)

        file_path = f"leaf{leaf_id:03}_shape_plot.png"
        if self.output_folder:
            file_path = os.path.join(self.output_folder, file_path)
        
        plt.figure(figsize=(12, 6))
        plt.plot(outline[0], outline[1], 'g-', linewidth=2)
        plt.fill(outline[0], outline[1], color='green', alpha=0.5)
        plt.ylabel("Length (inches)")  # Length on the y-axis
        plt.xlabel("Width (inches)")  # Width on the x-axis
        plt.title("Approximate Leaf Shape")
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(file_path)  

        return file_path, outline

if __name__ == "__main__":
    leafPlotter = LeafPlotter()

    file_path, outline = leafPlotter.Plot(2)
    print(file_path)

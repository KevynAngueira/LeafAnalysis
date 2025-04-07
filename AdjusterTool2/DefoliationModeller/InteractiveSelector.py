# Author: Kevyn Angueira Irizarry
# Created: 2025-04-07
# Last Modified: 2025-04-07

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_interactive_browning_selector(self, grid_cells):
    fig, ax = plt.subplots()

    # Plot leaf shape
    x, y = self.leaf_shape.exterior.xy
    ax.plot(x, y, color='green')

    # Draw grid and store rectangles
    cell_patches = []
    for cell in grid_cells:
        minx, miny, maxx, maxy = cell.bounds
        rect = Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=0.5, edgecolor='gray', facecolor='none')
        ax.add_patch(rect)
        cell_patches.append((rect, cell))

    selected_cells = []

    def on_click(event):
        for rect, cell in cell_patches:
            if rect.contains_point((event.xdata, event.ydata)):
                if cell in selected_cells:
                    selected_cells.remove(cell)
                    rect.set_facecolor('none')
                else:
                    selected_cells.append(cell)
                    rect.set_facecolor('brown')
                fig.canvas.draw()
                break

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return selected_cells

show_interactive_browning_selector()

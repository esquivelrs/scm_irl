import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

from matplotlib.colors import LinearSegmentedColormap




def plot_cropbox_area(depths_lands_inside, crop_box, env, color='blue'):
    
    for depth in depths_lands_inside:
        polygon = patches.Polygon([(x[0], x[1]) for x in depth[0].exterior.coords], fill=True, color=modulate_color(color, 1- depth[1]/100))
        plt.gca().add_patch(polygon)

    # add crop box
    polygon = patches.Polygon([(x[0], x[1]) for x in crop_box.exterior.coords], fill=False, color='red')
    plt.gca().add_patch(polygon)

    ellipse = Ellipse((env.agent_state.lon, env.agent_state.lat), 400, 200, fill=True, color='red', angle=90-(env.agent_state.cog)*180/np.pi)

    plt.gca().add_patch(ellipse)

    # Set the axes limits to fit the plot
    plt.xlim(env.scenario.east_min, env.scenario.east_max)
    plt.ylim(env.scenario.north_min, env.scenario.north_max)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def modulate_color(color, factor):
    # Convert the color to RGB
    rgb = mcolors.to_rgb(color)

    # Modulate each of the RGB values
    modulated_rgb = [x * factor for x in rgb]

    # Make sure each value is within the valid range [0, 1]
    modulated_rgb = [min(max(x, 0), 1) for x in modulated_rgb]

    return modulated_rgb


def cmap_seachart():
    color = 'dodgerblue'

    colors = [(0, "green")]
    for i in range(1, 256):
        colors.append((i/255, modulate_color(color, 1-i/255)))

    cmap = LinearSegmentedColormap.from_list('custom', colors)

    return cmap


def apply_cmap(matrix, cmap):
    # Normalize the matrix to the range [0, 1]
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    # Apply the color map
    matrix_rgb = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix_rgb[i, j] = cmap(matrix[i, j])[:3]

    # Convert the RGB matrix to an 8-bit matrix
    matrix_rgb = (matrix_rgb * 255).astype(np.uint8)

    return matrix_rgb

def create_color_map():
    color = (30, 144, 255)  # RGB for 'dodgerblue'
    colors = [(69, 137, 0)]  # RGB for 'green'
    for i in range(1, 256):
        colors.append((color[0], color[1] * (1 - i / 255), color[2] * (1 - i / 255)))
    return colors
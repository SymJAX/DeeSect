import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource


def meshgrid_mapping(
    mapping, *grids, return_grid=False, meshgrid=None,
):
    if meshgrid is None:
        meshgrid = np.meshgrid(*grids)

    meshgrid_flat = np.vstack([grid.reshape(-1) for grid in meshgrid]).T

    output = mapping(meshgrid_flat).reshape(meshgrid[0].shape)

    if return_grid == 2:
        return output, meshgrid, meshgrid_2d
    elif return_grid == 1:
        return output, meshgrid
    else:
        return output


def partition_boundary_2d(unit_values, grid):
    lines = []
    for k in range(h.shape[1]):
        a = plt.contour(
            grid[0], grid[1], unit_values[:, k].reshape(grid[0].shape), [0]
        )
        p = a.collections[0].get_paths()[0]
        lines.append(p.vertices)
        plt.close()
    return lines


def probe_2d_network(
    input_units_mapping,
    input_output_mapping,
    input_code_mapping=None,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-1, 1, -1, 1],
    meshgrid=None,
):

    if not meshgrid:
        grid_x = np.linspace(extent[0], extent[1], n_samples_x)
        grid_y = np.linspace(extent[2], extent[3], n_samples_y)

        unit_values, grid, grid_2d = meshgrid_mapping(
            input_units_mapping, grid_x, grid_y, return_grid=2,
        )
    else:
        unit_values, grid, grid_2d = meshgrid_mapping(
            input_units_mapping, return_grid=2, meshgrid=meshgrid
        )

    lines = partition_boundary_2d(unit_values, grid)

    all_points = np.concatenate(lines, 0)
    all_mapped_points = input_output_mapping(points)
    mapped_lines = []
    cpt = 0
    for line in lines:
        mapped_lines.append(all_mapped_points[cpt : cpt + len(line)])
        cpt += len(line)

    surface = input_output_mapping(grid_2d)

    if input_output_mapping:
        codes = input_code_mapping(grid_2d)
        return lines, mapped_lines, surface, grid, codes
    else:
        return lines, mapped_lines, surface, grid


def pretty_plot(
    input_units_mapping,
    input_output_mapping,
    input_code_mapping,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-1, 1, -1, 1],
    meshgrid=None,
):
    lines, mapped_lines, surface, grid, codes = probe_2d_1d_network(
        input_units_mapping,
        input_output_mapping,
        input_code_mapping,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y,
        extent=extent,
        meshgrid=meshgrid,
        return_grid=1,
    )

    unique_codes = set(map(tuple, codes))

    plt.figure()
    for line in lines:
        for start, end in zip(line[:-1], line[1:]):
            plt.plot(start, end)

    fig = plt.figure(figsize=(18, 12))

    # 0 degrees azimuth, 0 degrees elevation.
    light = LightSource(0, 0)

    # Generate face colors for a shaded surface using either
    # a color map or the uniform rgb color specified above.

    illuminated_surface = light.shade_rgb(red, Z)

    # Create a subplot with 3d plotting capabilities.
    # This command will fail if Axes3D was not imported.
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    # Set view parameters for all subplots.
    azimuth = 45
    altitude = 60
    ax.view_init(altitude, azimuth)
    ax.plot_surface(
        grid[0],
        grid[1],
        surface,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        facecolors=illuminated_surface,
    )

    for line in mapped_lines:
        for start, end in zip(line[:-1], line[1:]):
            ax.plot(start, end)
    plt.show()

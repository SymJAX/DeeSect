import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.ticker as mticker
import geometry


plt.style.use("/home/vrael/DeeSect/presentation.mplstyle")


def partition_boundary_2d(
    mapping,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-1, 1, -1, 1],
    meshgrid=None,
):

    if not meshgrid:
        grid_x = np.linspace(extent[0], extent[1], n_samples_x)
        grid_y = np.linspace(extent[2], extent[3], n_samples_y)
        meshgrid = np.meshgrid(grid_x, grid_y)

    meshgrid_flat = np.hstack([grid.reshape((-1, 1)) for grid in meshgrid])

    output = mapping(meshgrid_flat)

    if type(output) == list:
        for out in output:
            assert type(out) == np.ndarray
    else:
        assert type(output) == np.ndarray
        output = [output]

    paths = []
    for out in output:
        paths.append([])
        for k in range(out.shape[1]):
            a = plt.contour(
                *meshgrid,
                out[:, k].reshape(meshgrid[0].shape),
                [0],
            )
            unit_paths = a.collections[0].get_paths()
            for path in unit_paths:
                if len(path) > 1:
                    paths[-1].append(path)
            plt.close()
    return paths, meshgrid, meshgrid_flat


def make_3daxis_pretty(ax, xlabel="$x$", ylabel="$y$", azim=-110, elev=27):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ax.xaxis.pane.fill = False
    V = 250
    V2 = 250
    ax.w_xaxis.set_pane_color((V / 256, V / 256, V / 256, 0.8))
    ax.w_yaxis.set_pane_color((V / 256, V / 256, V / 256, 0.8))
    ax.w_zaxis.set_pane_color((V2 / 256, V2 / 256, V2 / 256, 0.8))
    # make the grid lines transparent
    C = 100
    ax.xaxis._axinfo["grid"]["color"] = (C / 256, C / 256, C / 256, 1)
    ax.yaxis._axinfo["grid"]["color"] = (C / 256, C / 256, C / 256, 1)
    ax.zaxis._axinfo["grid"]["color"] = (C / 256, C / 256, C / 256, 1)

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.set_zticks([])

    # tmp_planes = ax.zaxis._PLANES
    # ax.zaxis._PLANES = (
    #     tmp_planes[2],
    #     tmp_planes[3],
    #     tmp_planes[0],
    #     tmp_planes[1],
    #     tmp_planes[4],
    #     tmp_planes[5],
    # )

    ax.view_init(azim=azim, elev=elev)


def draw_layer_paths(ax, paths, layer_colors=["k"], **line_kwargs):
    if len(layer_colors) == 1:
        colors = layer_colors * len(paths)
    elif len(layer_colors) != len(paths):
        raise RuntimeError

    for layer, layerpath in enumerate(paths):
        for path in layerpath:
            if hasattr(path, "vertices"):
                ax.plot(
                    path.vertices[:-1, 0],
                    path.vertices[:-1, 1],
                    c=layer_colors[layer],
                    **line_kwargs,
                )
            elif path.shape[1] == 2:
                ax.plot(
                    path[:-1, 0],
                    path[:-1, 1],
                    c=layer_colors[layer],
                    **line_kwargs,
                )
            elif path.shape[1] == 3:
                ax.plot(
                    path[:-1, 0],
                    path[:-1, 1],
                    path[:-1, 2],
                    c=layer_colors[layer],
                    **line_kwargs,
                )


def codes_to_colors(codes, cmap="Spectral", n_colors=10):
    unique_codes = set(map(tuple, codes))
    unique_values = np.linspace(0, 1, min(n_colors, len(unique_codes)))[
        np.arange(len(unique_codes)) % n_colors
    ]
    code_to_value = {a: t for a, t in zip(unique_codes, unique_values)}

    cmap = matplotlib.cm.get_cmap(cmap)

    values = np.array([code_to_value[tuple(code)] for code in codes])
    colors = cmap(values)
    return values, colors


def draw_colored_partition_from_codes(
    ax, codes, xy_shape, cmap="Spectral", n_colors=10, **imshow_kwargs
):
    values, colors = codes_to_colors(codes, cmap="Spectral", n_colors=10)
    ax.imshow(
        values.reshape(xy_shape), cmap=cmap, vmin=0, vmax=1, **imshow_kwargs
    )

    return values, colors


def pretty_plot(
    per_unit_mapping,
    input_output_mapping=None,
    input_code_mapping=None,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-2, 2, -2, 2],
    meshgrid=None,
    n_colors=10,
    cmap="Spectral",
    color_values=None,
    color_mapping=None,
    name_input_space=None,
    name_output_space=None,
):
    paths, meshgrid, meshgrid_flat = partition_boundary_2d(
        per_unit_mapping,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y,
        extent=extent,
        meshgrid=meshgrid,
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if (
        color_values is None
        and color_mapping is None
        and input_code_mapping is not None
    ):
        codes = input_code_mapping(meshgrid_flat)
        values, colors = draw_colored_partition_from_codes(
            ax,
            codes,
            xy_shape=meshgrid[0].shape,
            n_colors=n_colors,
            cmap=cmap,
            extent=[
                meshgrid[0].min(),
                meshgrid[0].max(),
                meshgrid[1].min(),
                meshgrid[1].max(),
            ],
        )
    else:
        cmap = matplotlib.cm.get_cmap(cmap)

        if color_values is not None:
            values = color_values
        else:
            values = color_mapping(meshgrid_flat)

        colors = cmap(values)
        ax.pcolormesh(
            meshgrid[0],
            meshgrid[1],
            values.reshape(meshgrid[0].shape),
            cmap=cmap,
            alpha=1,
        )
    draw_layer_paths(ax, paths, zorder=1000)
    surface = input_output_mapping(meshgrid_flat)
    plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    if name_input_space is not None:
        plt.savefig(name_input_space)
        plt.close()

    mapped_paths = []
    for path in paths:
        mapped_paths.append([])
        for p in path:
            mapped_paths[-1].append(input_output_mapping(p.vertices[:-1]))

    fig = plt.figure()
    if surface.shape[1] == 1:
        ax = fig.add_subplot(1, 1, 1, projection="3d", proj_type="persp")
        ax.plot_surface(
            meshgrid[0],
            meshgrid[1],
            surface.reshape(meshgrid[0].shape),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            facecolors=colors.reshape(meshgrid[0].shape + (4,)),
            shade=True,
        )
        make_3daxis_pretty(ax)
    elif surface.shape[1] == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.pcolormesh(
            surface[:, 0].reshape(meshgrid[0].shape),
            surface[:, 1].reshape(meshgrid[0].shape),
            values.reshape(meshgrid[0].shape),
            cmap=cmap,
            alpha=1,
        )
        plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    elif surface.shape[1] == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d", proj_type="persp")
        ax.plot_surface(
            surface[:, 0].reshape(meshgrid[0].shape),
            surface[:, 1].reshape(meshgrid[0].shape),
            surface[:, 2].reshape(meshgrid[0].shape),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            facecolors=colors.reshape(meshgrid[0].shape + (4,)),
            shade=True,
        )
        make_3daxis_pretty(ax)
    draw_layer_paths(
        ax,
        mapped_paths,
        zorder=1000,
        linewidth=matplotlib.rcParams["lines.linewidth"] * 0.8,
    )

    if name_input_space is not None:
        plt.savefig(name_output_space)
        plt.close()

    plt.show()


def pretty_onelayer_partition(
    layer_W,
    layer_b,
    layer_alpha,
    n_samples_x=500,
    n_samples_y=500,
    extent=[-3, 3, -3, 3],
    meshgrid=None,
    n_colors=10,
    cmap="Spectral",
    name=None,
    with_power_diagram=True,
):
    def per_unit_mapping(x):
        h = x.dot(layer_W.T) + layer_b
        return h

    def input_code_mapping(x):
        h = x.dot(layer_W.T) + layer_b
        return (h > 0).astype("int32")

    paths, meshgrid, meshgrid_flat = partition_boundary_2d(
        per_unit_mapping,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y,
        extent=extent,
        meshgrid=meshgrid,
    )

    codes = input_code_mapping(meshgrid_flat)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    values, colors = draw_colored_partition_from_codes(
        ax,
        codes,
        xy_shape=meshgrid[0].shape,
        n_colors=n_colors,
        cmap=cmap,
        extent=[
            meshgrid[0].min(),
            meshgrid[0].max(),
            meshgrid[1].min(),
            meshgrid[1].max(),
        ],
    )

    draw_layer_paths(ax, paths, zorder=1000)

    if with_power_diagram:
        mus, radii, colors = geometry.get_layer_PD(
            meshgrid_flat, layer_W, layer_b, layer_alpha, colors
        )

        for center, rad, color in zip(mus, radii, colors):

            fc = color.copy()
            fc[-1] = 0.3
            fc[:-1] += 0.08
            fc = np.clip(fc, 0, 1)

            ec = color.copy()
            ec[-1] = 0.8
            ec[:-1] -= 0.08
            ec = np.clip(ec, 0, 1)

            circle = matplotlib.patches.Circle(
                center, radius=rad, facecolor=fc, edgecolor=ec, zorder=1500
            )
            ax.add_artist(circle)
            ax.scatter(*center, c=[color], zorder=2000, edgecolor="k")

    ax.set_xlim([meshgrid[0].min(), meshgrid[0].max()])
    ax.set_ylim([meshgrid[1].min(), meshgrid[1].max()])
    plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    if name is not None:
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name)
    else:
        plt.show()

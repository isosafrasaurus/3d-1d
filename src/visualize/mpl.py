import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from graphnics import *
from xii import *
from .util import *

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.75 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

def plot_3d_box(ax, box, color, label=None):
    lower, upper = box
    corners = np.array([
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]]
    ])
    for idx, (i, j) in enumerate(BOX_EDGES):
        kwargs = {'color': color}
        if idx == 0 and label:
            kwargs['label'] = label
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], **kwargs)

def plot_with_boundaries(uh1d, uh3d, z_level=None, cube_lower=None, cube_upper=None):
    mesh1d, mesh3d = uh1d.function_space().mesh(), uh3d.function_space().mesh()
    coords1d, coords3d = mesh1d.coordinates(), mesh3d.coordinates()
    values1d, values3d = uh1d.compute_vertex_values(mesh1d), uh3d.compute_vertex_values(mesh3d)

    z_level = z_level or np.median(coords3d[:, 2])
    fig = plt.figure(figsize=(14, 6))
    
    # 3D Scatter Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2], c=values1d, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax1, label='1D Pressure (Pa)')
    ax1.set(title='1D Pressure Scatter', xlabel='X', ylabel='Y', zlabel='Z')
    set_axes_equal(ax1)

    omega_box, sub_box = compute_boundaries(coords3d), compute_boundaries(coords1d)
    plot_3d_box(ax1, (list(omega_box[::2]), list(omega_box[1::2])), color='red', label='Omega Boundary')
    plot_3d_box(ax1, (list(sub_box[::2]), list(sub_box[1::2])), color='blue', label='Sub-mesh Boundary')
    if cube_lower:
        plot_3d_box(ax1, cube_lower, color='green', label='Lower Cube Subdomain')
    if cube_upper:
        plot_3d_box(ax1, cube_upper, color='magenta', label='Upper Cube Subdomain')
    ax1.legend()

    # 2D Heatmap Plot at Specified Z-Level
    tol = 1e-3
    mask = np.abs(coords3d[:, 2] - z_level) < tol
    if not mask.any():
        print(f"No data found at Z={z_level}")
        return fig  # Return figure even if no heatmap is created
    x_slice, y_slice, z_vals = coords3d[mask, 0], coords3d[mask, 1], values3d[mask]
    xi = np.linspace(x_slice.min(), x_slice.max(), 100)
    yi = np.linspace(y_slice.min(), y_slice.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_slice, y_slice), z_vals, (xi, yi), method='cubic')

    ax2 = fig.add_subplot(1, 2, 2)
    hm = ax2.imshow(zi, extent=(x_slice.min(), x_slice.max(),
                                y_slice.min(), y_slice.max()),
                    origin='lower', cmap='viridis')
    fig.colorbar(hm, ax=ax2, label='3D Pressure (Pa)')
    ax2.set(title=f'3D Pressure Heatmap at Z = {z_level:.3f}', xlabel='X', ylabel='Y')
    ax2.set_aspect('equal', adjustable='box')
    
    for box, color, label in zip([omega_box, sub_box], ['red', 'blue'], ['Omega Boundary', 'Sub-mesh Boundary']):
        rect = Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                         edgecolor=color, facecolor='none', lw=2, label=label)
        ax2.add_patch(rect)
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_path_pressure(uh1d, G, path):
    node_ids = get_cells_along_path(G, path)
    mesh = uh1d.function_space().mesh()
    coords = mesh.coordinates()
    pressure = uh1d.compute_vertex_values(mesh)
    path_coords, path_pressure = coords[node_ids], pressure[node_ids]
    
    print("Node Pressure Data:")
    for n, p in zip(node_ids, path_pressure):
        print(f"Node {n}: Pressure {p:.1f}")
        
    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(cum_dist, path_pressure, marker='o', markersize=2)
    plt.xlabel('Cumulative Distance Along Path (m)')
    plt.ylabel('1D Pressure (Pa)')
    plt.title('Pressure Along Path')
    plt.grid(True)
    return fig
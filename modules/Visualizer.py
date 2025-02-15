import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

def plot(uh1d, uh3d, z_level=None):
    mesh1d = uh1d.function_space().mesh()
    coords1d = mesh1d.coordinates()
    values1d = uh1d.compute_vertex_values(mesh1d)
    mesh3d = uh3d.function_space().mesh()
    coords3d = mesh3d.coordinates()
    values3d = uh3d.compute_vertex_values(mesh3d)

    if z_level is None:
        z_level = np.median(coords3d[:, 2])

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                     c=values1d, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax1, label='1D Pressure')
    ax1.set_title('1D Pressure Scatter')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    tol = 1e-3  # Tolerance for selecting the slice
    mask = np.abs(coords3d[:, 2] - z_level) < tol
    if not np.any(mask):
        print(f"No data found at Z={z_level}")
        return
    x = coords3d[mask, 0]
    y = coords3d[mask, 1]
    z = values3d[mask]

    # Create a grid for the heatmap
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data onto the grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Plot the heatmap
    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                         origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
    ax2.set_title(f'3D Pressure Heatmap at Z={z_level:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

def plot_with_boundaries(uh1d, uh3d, z_level=None, cube_lower=None, cube_upper=None):
    """
    Plots the 1D pressure scatter (in 3D) and a 2D heatmap of the 3D pressure
    at a given z-slice. On both diagrams, the boundaries of the overall 3D mesh (Omega)
    and the 1D sub-mesh are overlaid. In addition, only on the 3D plot the boundaries
    of the cube sub-domains (if provided) are drawn.
    
    Parameters:
      uh1d: FEniCS function defined on the 1D mesh (sub-mesh)
      uh3d: FEniCS function defined on the 3D mesh (Omega)
      z_level: z-coordinate for the 2D heatmap slice (if None, the median of Omega z-values is used)
      cube_lower: tuple (x_min, x_max, y_min, y_max, z_min, z_max) for the lower cube sub-domain
      cube_upper: tuple (x_min, x_max, y_min, y_max, z_min, z_max) for the upper cube sub-domain
    """
    # Retrieve 1D mesh data
    mesh1d = uh1d.function_space().mesh()
    coords1d = mesh1d.coordinates()
    values1d = uh1d.compute_vertex_values(mesh1d)

    # Retrieve 3D mesh data
    mesh3d = uh3d.function_space().mesh()
    coords3d = mesh3d.coordinates()
    values3d = uh3d.compute_vertex_values(mesh3d)

    # If no z_level is provided, take the median of the 3D z-coordinates.
    if z_level is None:
        z_level = np.median(coords3d[:, 2])

    # Create the figure with two subplots.
    fig = plt.figure(figsize=(14, 6))
    
    # ---------------------------
    # 3D Plot (left): 1D pressure scatter with boundaries.
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                     c=values1d, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax1, label='1D Pressure')
    ax1.set_title('1D Pressure Scatter (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Helper function to plot a 3D bounding box.
    def plot_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color, label=None):
        corners = np.array([[x_min, y_min, z_min],
                            [x_max, y_min, z_min],
                            [x_max, y_max, z_min],
                            [x_min, y_max, z_min],
                            [x_min, y_min, z_max],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_max],
                            [x_min, y_max, z_max]])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                 (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # vertical edges
        first_edge = True
        for i, j in edges:
            if first_edge and label is not None:
                ax.plot([corners[i, 0], corners[j, 0]],
                        [corners[i, 1], corners[j, 1]],
                        [corners[i, 2], corners[j, 2]],
                        color=color, label=label)
                first_edge = False
            else:
                ax.plot([corners[i, 0], corners[j, 0]],
                        [corners[i, 1], corners[j, 1]],
                        [corners[i, 2], corners[j, 2]],
                        color=color)

    # Compute the overall boundary for Omega (3D mesh)
    omega_x_min, omega_x_max = coords3d[:, 0].min(), coords3d[:, 0].max()
    omega_y_min, omega_y_max = coords3d[:, 1].min(), coords3d[:, 1].max()
    omega_z_min, omega_z_max = coords3d[:, 2].min(), coords3d[:, 2].max()

    # Compute the bounding box for the 1D sub-mesh (projected in 3D)
    sub_x_min, sub_x_max = coords1d[:, 0].min(), coords1d[:, 0].max()
    sub_y_min, sub_y_max = coords1d[:, 1].min(), coords1d[:, 1].max()
    sub_z_min, sub_z_max = coords1d[:, 2].min(), coords1d[:, 2].max()

    # Plot the Omega boundary and the sub-mesh boundary on the 3D plot.
    plot_3d_box(ax1, omega_x_min, omega_x_max, omega_y_min, omega_y_max, omega_z_min, omega_z_max,
                color='red', label='Omega Boundary')
    plot_3d_box(ax1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, sub_z_min, sub_z_max,
                color='blue', label='Sub-mesh Boundary')

    # If cube sub-domain boundaries are provided, overlay them (only in 3D).
    if cube_lower is not None:
        # cube_lower = (x_min, x_max, y_min, y_max, z_min, z_max)
        plot_3d_box(ax1, cube_lower[0], cube_lower[1],
                    cube_lower[2], cube_lower[3],
                    cube_lower[4], cube_lower[5],
                    color='green', label='Lower Cube Subdomain')
    if cube_upper is not None:
        plot_3d_box(ax1, cube_upper[0], cube_upper[1],
                    cube_upper[2], cube_upper[3],
                    cube_upper[4], cube_upper[5],
                    color='magenta', label='Upper Cube Subdomain')

    ax1.legend()

    # ---------------------------
    # 2D Plot (right): 3D pressure heatmap at a chosen z-level.
    tol = 1e-3  # tolerance for selecting points at the desired z-level
    mask = np.abs(coords3d[:, 2] - z_level) < tol
    if not np.any(mask):
        print(f"No data found at Z={z_level}")
        return
    x_slice = coords3d[mask, 0]
    y_slice = coords3d[mask, 1]
    z_slice_values = values3d[mask]

    # Create a grid for interpolation.
    xi = np.linspace(x_slice.min(), x_slice.max(), 100)
    yi = np.linspace(y_slice.min(), y_slice.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_slice, y_slice), z_slice_values, (xi, yi), method='cubic')

    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(zi, extent=(x_slice.min(), x_slice.max(),
                                      y_slice.min(), y_slice.max()),
                         origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
    ax2.set_title(f'3D Pressure Heatmap at Z = {z_level:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Overlay the Omega and sub-mesh boundaries on the 2D heatmap.
    # (Cube sub-domain boundaries are shown only in the 3D plot.)
    omega_rect = Rectangle((omega_x_min, omega_y_min),
                           omega_x_max - omega_x_min,
                           omega_y_max - omega_y_min,
                           edgecolor='red', facecolor='none', lw=2, label='Omega Boundary')
    ax2.add_patch(omega_rect)
    sub_rect = Rectangle((sub_x_min, sub_y_min),
                         sub_x_max - sub_x_min,
                         sub_y_max - sub_y_min,
                         edgecolor='blue', facecolor='none', lw=2, label='Sub-mesh Boundary')
    ax2.add_patch(sub_rect)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_path_pressure(uh1d, node_ids):
    # Get the mesh and coordinates from the 1D function space.
    mesh = uh1d.function_space().mesh()
    coords = mesh.coordinates()  # Array of shape (N, dim)
    
    # Get the pressure values at all mesh vertices.
    pressure_values = uh1d.compute_vertex_values(mesh)
    
    # Extract the coordinates and pressure values along the provided path.
    # It is assumed that the node_ids correspond directly to the row indices of coords.
    path_coords = np.array([coords[i] for i in node_ids])
    path_pressure = np.array([pressure_values[i] for i in node_ids])

    print("Node Pressure Data:")
    for node, p in zip(node_ids, path_pressure):
        print(f"Node {node}: Pressure {p:.1f}")
    
    # Calculate cumulative distances along the path.
    cumulative_distance = [0.0]
    for i in range(1, len(path_coords)):
        # Euclidean distance between consecutive nodes
        d = np.linalg.norm(path_coords[i] - path_coords[i - 1])
        cumulative_distance.append(cumulative_distance[-1] + d)
    cumulative_distance = np.array(cumulative_distance)
    
    # Plotting the 1D pressure along the path.
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_distance, path_pressure, marker='o', linestyle='-', markersize=5)
    plt.xlabel('Cumulative Distance Along Path (m)')
    plt.ylabel('1D Pressure (Pa)')
    plt.title('Pressure Along Path')
    plt.grid(True)
    
    plt.show()
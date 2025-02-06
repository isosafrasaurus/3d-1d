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
    
    # Annotate each point with its pressure value, alternating label positions.
    for i, (x, p) in enumerate(zip(cumulative_distance, path_pressure)):
        # Alternate the vertical offset: above for even indices, below for odd.
        if i % 2 == 0:
            offset = (0, 10)
        else:
            offset = (0, -10)
        
        plt.annotate(
            f"{p:.1f}",
            xy=(x, p),
            xytext=offset,
            textcoords='offset points',
            ha='center',
            color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
        )
    
    plt.show()
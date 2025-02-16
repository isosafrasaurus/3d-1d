import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from graphnics import *
from xii import *
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

def plot_with_boundaries(uh1d, uh3d, z_level=None, cube_lower=None, cube_upper=None):
    mesh1d = uh1d.function_space().mesh()
    coords1d = mesh1d.coordinates()
    values1d = uh1d.compute_vertex_values(mesh1d)

    mesh3d = uh3d.function_space().mesh()
    coords3d = mesh3d.coordinates()
    values3d = uh3d.compute_vertex_values(mesh3d)

    if z_level is None:
        z_level = np.median(coords3d[:, 2])

    fig = plt.figure(figsize=(14, 6))
    
    # 3D Plot: 1D pressure scatter with boundaries.
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                     c=values1d, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax1, label='1D Pressure')
    ax1.set_title('1D Pressure Scatter (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])
    
    def plot_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color, label=None):
        corners = np.array([[x_min, y_min, z_min],
                            [x_max, y_min, z_min],
                            [x_max, y_max, z_min],
                            [x_min, y_max, z_min],
                            [x_min, y_min, z_max],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_max],
                            [x_min, y_max, z_max]])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
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

    omega_x_min, omega_x_max = coords3d[:, 0].min(), coords3d[:, 0].max()
    omega_y_min, omega_y_max = coords3d[:, 1].min(), coords3d[:, 1].max()
    omega_z_min, omega_z_max = coords3d[:, 2].min(), coords3d[:, 2].max()

    sub_x_min, sub_x_max = coords1d[:, 0].min(), coords1d[:, 0].max()
    sub_y_min, sub_y_max = coords1d[:, 1].min(), coords1d[:, 1].max()
    sub_z_min, sub_z_max = coords1d[:, 2].min(), coords1d[:, 2].max()

    plot_3d_box(ax1, omega_x_min, omega_x_max, omega_y_min, omega_y_max, omega_z_min, omega_z_max,
                color='red', label='Omega Boundary')
    plot_3d_box(ax1, sub_x_min, sub_x_max, sub_y_min, sub_y_max, sub_z_min, sub_z_max,
                color='blue', label='Sub-mesh Boundary')

    if cube_lower is not None:
        # cube_lower = ([x_min, y_min, z_min], [x_max, y_max, z_max])
        plot_3d_box(ax1, cube_lower[0][0], cube_lower[1][0],
                    cube_lower[0][1], cube_lower[1][1],
                    cube_lower[0][2], cube_lower[1][2],
                    color='green', label='Lower Cube Subdomain')
    if cube_upper is not None:
        plot_3d_box(ax1, cube_upper[0][0], cube_upper[1][0],
                    cube_upper[0][1], cube_upper[1][1],
                    cube_upper[0][2], cube_upper[1][2],
                    color='magenta', label='Upper Cube Subdomain')

    ax1.legend()

    # 2D Plot: 3D pressure heatmap at a chosen z-level.
    tol = 1e-3 
    mask = np.abs(coords3d[:, 2] - z_level) < tol
    if not np.any(mask):
        print(f"No data found at Z={z_level}")
        return
    x_slice = coords3d[mask, 0]
    y_slice = coords3d[mask, 1]
    z_slice_values = values3d[mask]

    xi = np.linspace(x_slice.min(), x_slice.max(), 100)
    yi = np.linspace(y_slice.min(), y_slice.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_slice, y_slice), z_slice_values, (xi, yi), method='cubic')

    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(zi, extent=(x_slice.min(), x_slice.max(),
                                      y_slice.min(), y_slice.max()),
                         origin='lower', cmap='viridis')
    fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
    ax2.set_title(f'3D Pressure Heatmap at Z = {z_level:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    ax2.set_aspect('equal', adjustable='box')
    
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

def plot_path_pressure(uh1d, G, path):
    node_ids = get_cells_along_path(G, path)
    
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
    plt.plot(cumulative_distance, path_pressure, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Cumulative Distance Along Path (m)')
    plt.ylabel('1D Pressure (Pa)')
    plt.title('Pressure Along Path')
    plt.grid(True)
    
    plt.show()

def get_cells_along_path(G, path):
	"""
	Given a FenicsGraph G and a path (list of node IDs in G),
	return the ordered list of global vertex IDs (from G.mesh)
	that lie along that path.
	"""
	global_vertices = []
	
	for i in range(len(path) - 1):
		u = path[i]
		v = path[i+1]
		
		# Check if edge exists in forward or reverse direction.
		if G.has_edge(u, v):
			edge = (u, v)
			forward = True
		elif G.has_edge(v, u):
			edge = (v, u)
			forward = False
		else:
			raise ValueError(f"No edge between {u} and {v} in the graph.")
			
		# Retrieve the submesh for this edge.
		submesh = G.edges[edge]["submesh"]
		coords = submesh.coordinates()  # shape (n_vertices, geom_dim)
		
		# Try to obtain the mapping from local to global vertex indices.
		if hasattr(submesh, 'entity_map'):
			local_to_global = submesh.entity_map(0)
		else:
			# Fall back to matching coordinates.
			global_coords = G.mesh.coordinates()
			tol = 1e-12
			local_to_global = []
			for local_pt in coords:
				matches = np.where(np.all(np.isclose(global_coords, local_pt, atol=tol), axis=1))[0]
				if len(matches) == 0:
					raise ValueError("No matching global vertex found for local coordinate: " + str(local_pt))
				local_to_global.append(matches[0])
			local_to_global = np.array(local_to_global)
			
		# Determine the correct tangent for ordering.
		tangent = G.edges[edge]["tangent"]
		if not forward:
			tangent = -tangent
			
		# Project the submesh vertex coordinates onto the tangent.
		proj = np.dot(coords, tangent)
		sorted_local_indices = np.argsort(proj)
		
		# Map the sorted local indices to global vertex IDs.
		ordered_globals = [local_to_global[idx] for idx in sorted_local_indices]
		
		# Avoid duplicate vertices at interfaces.
		if i > 0 and ordered_globals[0] == global_vertices[-1]:
			ordered_globals = ordered_globals[1:]
			
		global_vertices.extend(ordered_globals)
		
	return global_vertices
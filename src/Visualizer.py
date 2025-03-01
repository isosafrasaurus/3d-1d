import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List, Any

from graphnics import *
from xii import *

def set_axes_equal(ax: Any) -> None:
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_middle, y_middle, z_middle = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    plot_radius = 0.75 * max(abs(x_limits[1] - x_limits[0]),
                             abs(y_limits[1] - y_limits[0]),
                             abs(z_limits[1] - z_limits[0]))
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_box_edges(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    x_line, y_line, z_line = [], [], []
    for i, j in edges:
        x_line.extend([corners[i, 0], corners[j, 0], None])
        y_line.extend([corners[i, 1], corners[j, 1], None])
        z_line.extend([corners[i, 2], corners[j, 2], None])
    return np.array(x_line), np.array(y_line), np.array(z_line)

def plot_3d_box(ax: Any, box: Tuple[np.ndarray, np.ndarray], color: str, label: Optional[str] = None) -> None:
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
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    first_edge = True
    for i, j in edges:
        kwargs = {'color': color}
        if first_edge and label:
            kwargs['label'] = label
            first_edge = False
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                **kwargs)

def compute_boundaries(coords: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    return (coords[:, 0].min(), coords[:, 0].max(),
            coords[:, 1].min(), coords[:, 1].max(),
            coords[:, 2].min(), coords[:, 2].max())

def plot_with_boundaries(uh1d: Any, uh3d: Any,
                         z_level: Optional[float] = None,
                         cube_lower: Optional[Tuple[List[float], List[float]]] = None,
                         cube_upper: Optional[Tuple[List[float], List[float]]] = None) -> None:
    mesh1d, mesh3d = uh1d.function_space().mesh(), uh3d.function_space().mesh()
    coords1d, coords3d = mesh1d.coordinates(), mesh3d.coordinates()
    values1d, values3d = uh1d.compute_vertex_values(mesh1d), uh3d.compute_vertex_values(mesh3d)

    if z_level is None:
        z_level = np.median(coords3d[:, 2])

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                     c=values1d, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax1, label='1D Pressure (Pa)')
    ax1.set_title('1D Pressure Scatter')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)

    omega_box = compute_boundaries(coords3d)
    sub_box = compute_boundaries(coords1d)

    plot_3d_box(ax1, ([omega_box[0], omega_box[2], omega_box[4]],
                        [omega_box[1], omega_box[3], omega_box[5]]),
                color='red', label='Omega Boundary')
    plot_3d_box(ax1, ([sub_box[0], sub_box[2], sub_box[4]],
                        [sub_box[1], sub_box[3], sub_box[5]]),
                color='blue', label='Sub-mesh Boundary')
    if cube_lower is not None:
        plot_3d_box(ax1, cube_lower, color='green', label='Lower Cube Subdomain')
    if cube_upper is not None:
        plot_3d_box(ax1, cube_upper, color='magenta', label='Upper Cube Subdomain')
    ax1.legend()

    tol = 1e-3
    mask = np.abs(coords3d[:, 2] - z_level) < tol
    if not np.any(mask):
        print(f"No data found at Z={z_level}")
        return

    x_slice, y_slice, z_slice_vals = coords3d[mask, 0], coords3d[mask, 1], values3d[mask]
    xi = np.linspace(x_slice.min(), x_slice.max(), 100)
    yi = np.linspace(y_slice.min(), y_slice.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x_slice, y_slice), z_slice_vals, (xi, yi), method='cubic')

    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(zi, extent=(x_slice.min(), x_slice.max(),
                                       y_slice.min(), y_slice.max()),
                         origin='lower', cmap='viridis')
    fig.colorbar(heatmap, ax=ax2, label='3D Pressure (Pa)')
    ax2.set_title(f'3D Pressure Heatmap at Z = {z_level:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal', adjustable='box')

    omega_rect = Rectangle((omega_box[0], omega_box[2]),
                           omega_box[1] - omega_box[0],
                           omega_box[3] - omega_box[2],
                           edgecolor='red', facecolor='none', lw=2, label='Omega Boundary')
    sub_rect = Rectangle((sub_box[0], sub_box[2]),
                         sub_box[1] - sub_box[0],
                         sub_box[3] - sub_box[2],
                         edgecolor='blue', facecolor='none', lw=2, label='Sub-mesh Boundary')
    ax2.add_patch(omega_rect)
    ax2.add_patch(sub_rect)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_cells_along_path(G: Any, path: List[int]) -> List[int]:
    global_vertices = []
    global_coords = G.mesh.coordinates()

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            edge, forward = (u, v), True
        elif G.has_edge(v, u):
            edge, forward = (v, u), False
        else:
            raise ValueError(f"No edge between {u} and {v} in the graph.")

        submesh = G.edges[edge]["submesh"]
        coords = submesh.coordinates()
        if hasattr(submesh, 'entity_map'):
            local_to_global = submesh.entity_map(0)
        else:
            tol = 1e-12
            local_to_global = []
            for local_pt in coords:
                matches = np.where(np.all(np.isclose(global_coords, local_pt, atol=tol), axis=1))[0]
                if len(matches) == 0:
                    raise ValueError(f"No matching global vertex for local coordinate: {local_pt}")
                local_to_global.append(matches[0])
            local_to_global = np.array(local_to_global)

        tangent = G.edges[edge]["tangent"]
        if not forward:
            tangent = -tangent
        proj = np.dot(coords, tangent)
        sorted_local_indices = np.argsort(proj)
        ordered_globals = [local_to_global[idx] for idx in sorted_local_indices]

        if i > 0 and ordered_globals[0] == global_vertices[-1]:
            ordered_globals = ordered_globals[1:]
        global_vertices.extend(ordered_globals)
    return global_vertices


def plot_path_pressure(uh1d: Any, G: Any, path: List[int]) -> None:
    node_ids = get_cells_along_path(G, path)
    mesh = uh1d.function_space().mesh()
    coords = mesh.coordinates()
    pressure_values = uh1d.compute_vertex_values(mesh)

    path_coords = np.array([coords[i] for i in node_ids])
    path_pressure = np.array([pressure_values[i] for i in node_ids])

    print("Node Pressure Data:")
    for node, p in zip(node_ids, path_pressure):
        print(f"Node {node}: Pressure {p:.1f}")

    cumulative_distance = np.cumsum(
        np.r_[0, np.linalg.norm(np.diff(path_coords, axis=0), axis=1)]
    )

    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_distance, path_pressure, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Cumulative Distance Along Path (m)')
    plt.ylabel('1D Pressure (Pa)')
    plt.title('Pressure Along Path')
    plt.grid(True)
    plt.show()


def add_box_plotly(fig: go.Figure,
                   box: Tuple[np.ndarray, np.ndarray],
                   color: str, name: str) -> None:
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
    x_line, y_line, z_line = get_box_edges(corners)
    fig.add_trace(go.Scatter3d(
        x=x_line, y=y_line, z=z_line,
        mode='lines',
        line=dict(color=color, width=4),
        name=name
    ))


def plot_with_boundaries_plotly(uh1d: Any, uh3d: Any,
                                cube_lower: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                cube_upper: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    mesh1d = uh1d.function_space().mesh()
    coords1d = mesh1d.coordinates()
    values1d = uh1d.compute_vertex_values(mesh1d)
    mesh3d = uh3d.function_space().mesh()
    coords3d = mesh3d.coordinates()

    omega_box = compute_boundaries(coords3d)
    sub_box = compute_boundaries(coords1d)

    fig = go.Figure()

    annotations = [f"({x:.2f}, {y:.2f}, {z:.2f})<br>P: {p:.2e}"
                   for (x, y, z), p in zip(coords1d, values1d)]

    fig.add_trace(go.Scatter3d(
        x=coords1d[:, 0],
        y=coords1d[:, 1],
        z=coords1d[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=values1d,
            colorscale='Viridis',
            colorbar=dict(title='1D Pressure (Pa)'),
            opacity=0.8
        ),
        text=annotations,
        textposition="top center",
        name='1D Pressure'
    ))

    add_box_plotly(fig, ([omega_box[0], omega_box[2], omega_box[4]],
                           [omega_box[1], omega_box[3], omega_box[5]]),
                   color='red', name='Omega Boundary')
    add_box_plotly(fig, ([sub_box[0], sub_box[2], sub_box[4]],
                           [sub_box[1], sub_box[3], sub_box[5]]),
                   color='blue', name='Sub-mesh Boundary')
    if cube_lower is not None:
        add_box_plotly(fig, cube_lower, color='green', name='Lower Cube Subdomain')
    if cube_upper is not None:
        add_box_plotly(fig, cube_upper, color='magenta', name='Upper Cube Subdomain')

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='1D Pressure Scatter with Boundaries',
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=700
    )
    fig.show()
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List, Any
from graphnics import *
from xii import *

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
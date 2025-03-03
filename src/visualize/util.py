import numpy as np
from graphnics import *

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

def get_box_edges(corners):
    x_line, y_line, z_line = [], [], []
    for i, j in BOX_EDGES:
        x_line += [corners[i, 0], corners[j, 0], None]
        y_line += [corners[i, 1], corners[j, 1], None]
        z_line += [corners[i, 2], corners[j, 2], None]
    return np.array(x_line), np.array(y_line), np.array(z_line)

def compute_boundaries(coords):
    return (coords[:, 0].min(), coords[:, 0].max(),
            coords[:, 1].min(), coords[:, 1].max(),
            coords[:, 2].min(), coords[:, 2].max())

def get_cells_along_path(G, path):
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
from typing import Optional, List
import numpy as np
from dolfin import UnitCubeMesh, MeshFunction, UserExpression
from graphnics import FenicsGraph
from rtree import index as rtree_index

class MeshCreator:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 0.008,
        Lambda_num_nodes_exp: int = 5,
    ):
        # Build and extract Lambda
        G.make_mesh(n=Lambda_num_nodes_exp)
        G.make_submeshes()
        self.Lambda, edge_marker = G.get_mesh(n=Lambda_num_nodes_exp)
        Lambda_coords = self.Lambda.coordinates()
        self.G_ref = G.copy()

        # Translate Lambda_coords to >= 0
        shift = -np.min(Lambda_coords, axis=0)
        Lambda_coords[:] += shift
        self.__shift_graph_nodes(self.G_ref, shift)

        # If Omega bounds are provided, recenter Lambda relative to Omega
        if Omega_bounds_dim is not None:
            lower, upper = np.array(Omega_bounds_dim[0]), np.array(Omega_bounds_dim[1])
            omega_center = (lower + upper) / 2
            lambda_center = np.mean(Lambda_coords, axis=0)
            center_shift = omega_center - lambda_center
            Lambda_coords[:] += center_shift
            self.__shift_graph_nodes(self.G_ref, center_shift)

        # Build and extract Omega
        self.Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = self.Omega.coordinates()

        if Omega_bounds_dim is None:
            min_lambda = np.min(Lambda_coords, axis=0)
            max_lambda = np.max(Lambda_coords, axis=0)
            scales = (max_lambda - min_lambda) + 2 * Lambda_padding_min
            shift_correction = Lambda_padding_min - min_lambda
            Lambda_coords[:] += shift_correction
            self.__shift_graph_nodes(self.G_ref, shift_correction)
            shifts = np.zeros(3)
        else:
            lower, upper = np.array(Omega_bounds_dim[0]), np.array(Omega_bounds_dim[1])
            scales = upper - lower
            shifts = lower

        Omega_coords[:] = Omega_coords * scales + shifts
        self.Omega_bounds = [shifts, shifts + scales]
        self.radius_map = self.radius_map(self.G_ref, edge_marker)

    @staticmethod
    def __shift_graph_nodes(G, shift: np.ndarray):
        for n in G.nodes:
            pos = np.array(G.nodes[n]['pos'])
            G.nodes[n]['pos'] = (pos + shift).tolist()

    class radius_map(UserExpression):
        def __init__(self, G: FenicsGraph, edge_marker: MeshFunction, **kwargs):
            super().__init__(**kwargs)
            self.G = G
            self.edge_marker = edge_marker

            # Create an R-tree spatial index in 3D
            p = rtree_index.Property()
            p.dimension = 3
            self.spatial_idx = rtree_index.Index(properties=p)
            self.edge_data_list = []

            for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
                pos_u = np.array(G.nodes[u]['pos'])
                pos_v = np.array(G.nodes[v]['pos'])
                radius = data['radius']
                bbox = tuple(np.concatenate([
                    np.minimum(pos_u, pos_v) - radius,
                    np.maximum(pos_u, pos_v) + radius
                ]).tolist())
                self.spatial_idx.insert(edge_id, bbox)
                self.edge_data_list.append((u, v, data))

        def eval(self, value, x):
            point = tuple(x[:3])
            # Query the spatial index for candidate edges
            candidates = self.spatial_idx.intersection(point * 2, objects=False)
            for edge_id in candidates:
                u, v, data = self.edge_data_list[edge_id]
                pos_u = np.array(self.G.nodes[u]['pos'])
                pos_v = np.array(self.G.nodes[v]['pos'])
                if self.point_in_cylinder(point, pos_u, pos_v, data['radius']):
                    value[0] = data['radius']
                    return
            value[0] = 0.0

        def value_shape(self):
            return ()

        @staticmethod
        def point_in_cylinder(point, pos_u, pos_v, radius):
            """Check if the point is within the cylinder defined by pos_u, pos_v, and radius."""
            p = np.array(point)
            u = np.array(pos_u)
            v = np.array(pos_v)
            line = v - u
            line_length_sq = np.dot(line, line)
            if line_length_sq == 0:
                return np.linalg.norm(p - u) <= radius

            t = np.dot(p - u, line) / line_length_sq
            t = np.clip(t, 0.0, 1.0)
            projection = u + t * line
            return np.linalg.norm(p - projection) <= radius

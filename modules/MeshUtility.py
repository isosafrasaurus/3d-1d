from typing import Optional, List
from dolfin import *
from graphnics import *
from xii import *
import networkx as nx
import numpy as np
from rtree import index as rtree_index

class MeshUtility:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 0.008,
        Lambda_num_nodes_exp: int = 5,
    ):
        # Create the mesh and submeshes using G
        G.make_mesh(n=Lambda_num_nodes_exp)
        G.make_submeshes()
        Lambda, Lambda_edge_marker = G.get_mesh(n=Lambda_num_nodes_exp)

        Lambda_coords = Lambda.coordinates()

        # Create the background Omega mesh and transform it according to Lambda's extents.
        Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = Omega.coordinates()

        if Omega_bounds_dim is None:
            xmin, ymin, zmin = np.min(Lambda_coords, axis=0)
            xmax, ymax, zmax = np.max(Lambda_coords, axis=0)
            scales = np.array([
                xmax - xmin + 2 * Lambda_padding_min,
                ymax - ymin + 2 * Lambda_padding_min,
                zmax - zmin + 2 * Lambda_padding_min
            ])
            shifts = np.array([
                xmin - Lambda_padding_min,
                ymin - Lambda_padding_min,
                zmin - Lambda_padding_min
            ])
        else:
            lower = np.array(Omega_bounds_dim[0])
            upper = np.array(Omega_bounds_dim[1])
            scales = upper - lower
            shifts = lower

        Omega_coords[:, :] = Omega_coords * scales + shifts

        self.Omega_bounds = [shifts, shifts + scales]
        self.Omega = Omega
        self.Lambda = Lambda
        self.Lambda_edge_marker = Lambda_edge_marker
        self.radius_map = self.radius_map(G, Lambda_edge_marker)

    class radius_map(UserExpression):
        def __init__(self, G: "FenicsGraph", edge_marker: MeshFunction, **kwargs):
            super().__init__(**kwargs)
            
            # Set up the R-tree for spatial indexing.
            p = rtree_index.Property()
            p.dimension = 3
            self.spatial_idx = rtree_index.Index(properties=p)
            self.edge_data_list = []

            # Iterate over graph edges and insert bounding boxes.
            for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
                pos_u = np.array(G.nodes[u]['pos'])
                pos_v = np.array(G.nodes[v]['pos'])
                radius = data['radius']

                min_coords = np.minimum(pos_u, pos_v) - radius
                max_coords = np.maximum(pos_u, pos_v) + radius

                # R-tree expects a bounding box in the form:
                # (minx, miny, minz, maxx, maxy, maxz)
                bbox = tuple(min_coords.tolist() + max_coords.tolist())
                self.spatial_idx.insert(edge_id, bbox)
                self.edge_data_list.append((u, v, data))

            self.G = G
            self.edge_marker = edge_marker

        def eval(self, value, x):
            # Get the point from x (assuming x has at least 3 components)
            point = tuple(x[:3])
            # Query the spatial index for candidate edges
            candidates = list(self.spatial_idx.intersection(point + point, objects=False))

            for edge_id in candidates:
                u, v, data = self.edge_data_list[edge_id]
                pos_u = np.array(self.G.nodes[u]['pos'])
                pos_v = np.array(self.G.nodes[v]['pos'])
                radius = data['radius']

                if self.point_in_cylinder(point, pos_u, pos_v, radius):
                    value[0] = radius
                    return

            value[0] = 0.0

        def value_shape(self):
            return ()

        @staticmethod
        def point_in_cylinder(point, pos_u, pos_v, radius):
            p = np.array(point)
            u = np.array(pos_u)
            v = np.array(pos_v)
            line = v - u
            line_length_sq = np.dot(line, line)
            if line_length_sq == 0:
                # The edge degenerates to a point.
                return np.linalg.norm(p - u) <= radius

            # Find the projection parameter t and clamp it to [0, 1]
            t = np.dot(p - u, line) / line_length_sq
            t = np.clip(t, 0.0, 1.0)
            projection = u + t * line
            distance = np.linalg.norm(p - projection)
            return distance <= radius
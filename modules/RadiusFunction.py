from dolfin import *
from graphnics import FenicsGraph
from xii import *
from rtree import index as rtree_index
import numpy as np

class RadiusFunction(UserExpression):
    def __init__(self, G: "FenicsGraph", edge_marker: MeshFunction, **kwargs):
        super().__init__(**kwargs)
        
        p = rtree_index.Property()
        p.dimension = 3
        spatial_idx = rtree_index.Index(properties=p)
        edge_data_list = []

        for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
            pos_u = np.array(G.nodes[u]['pos'])
            pos_v = np.array(G.nodes[v]['pos'])
            radius = data['radius']

            min_coords = np.minimum(pos_u, pos_v) - radius
            max_coords = np.maximum(pos_u, pos_v) + radius

            # R-tree expects bounding boxes in the form (minx, miny, minz, maxx, maxy, maxz)
            bbox = tuple(min_coords.tolist() + max_coords.tolist())
            spatial_idx.insert(edge_id, bbox)
            edge_data_list.append((u, v, data))

        self.G = G
        self.edge_marker = edge_marker
        self.spatial_idx = spatial_idx
        self.edge_data_list = edge_data_list

    def eval(self, value, x):
        point = tuple(x[:3])
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
            # Edge is actually a point
            return np.linalg.norm(p - u) <= radius

        # Parametric coordinate t in [0, 1] along the line from u to v
        t = np.dot(p - u, line) / line_length_sq
        t = np.clip(t, 0.0, 1.0)
        projection = u + t * line
        distance = np.linalg.norm(p - projection)
        return distance <= radius

import numpy as np
from typing import Optional, Tuple
from dolfin import UnitCubeMesh, MeshFunction, UserExpression, SubDomain, BoundingBoxTree, near, Point
from graphnics import FenicsGraph
from rtree import index as rtree_index

class AxisPlane(SubDomain):
    def __init__(self, axis, coordinate: float):
        super().__init__()
        self.axis = axis
        self.coordinate = coordinate

    def inside(self, x, on_boundary: bool) -> bool:
        return on_boundary and near(x[self.axis], self.coordinate)

# class RadiusMap(UserExpression):
#     def __init__(self, G, Lambda):
#         super().__init__()
#         self.G = G
#         self.tree = BoundingBoxTree()
#         self.tree.build(Lambda)
    
#     def eval(self, value, x):
#         p = Point(x[0], x[1], x[2])
#         cell = self.tree.compute_first_entity_collision(p)
#         edge_ix = self.G.mf[cell]
#         edge = list(self.G.edges())[edge_ix]
#         value[0] = self.G.edges()[edge]['radius']

class RadiusMap(UserExpression):
    def __init__(self, G, edge_marker):
        super().__init__()
        self.G = G
        self.edge_marker = edge_marker

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

class MeshBuild:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds: Optional[np.ndarray] = None,
        Omega_mesh_voxel_dim: Tuple[int, int, int] = (16, 16, 16),
        Lambda_padding: Optional[float] = 0.008,
        Lambda_num_nodes_exp: Optional[int] = 5
    ):
        assert Omega_bounds.shape == (2, 3), "Omega_bounds must have shape (2, 3) for span of Omega box"
        
        G.make_mesh(n = Lambda_num_nodes_exp); G.make_submeshes()
        self.Lambda, edge_marker = G.get_mesh(n = Lambda_num_nodes_exp)
        Lambda_coords = self.Lambda.coordinates()
        lambda_min = np.min(Lambda_coords, axis=0)
        lambda_max = np.max(Lambda_coords, axis=0)
        
        self.Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = self.Omega.coordinates()

        if Omega_bounds is None:
            scales = lambda_max - lambda_min + 2 * Lambda_padding
            shifts = lambda_min - Lambda_padding
            self.Omega_bounds = np.array([shifts.tolist(), (shifts + scales).tolist()])
        else:
            lower = np.minimum(Omega_bounds[0], Omega_bounds[1])
            upper = np.maximum(Omega_bounds[0], Omega_bounds[1])
            assert np.all(lambda_min >= lower) and np.all(lambda_max <= upper), (
                "Lambda mesh is not contained within Omega_bounds."
            )
            scales = upper - lower
            shifts = lower
            self.Omega_bounds = np.vstack((lower, upper))

        Omega_coords[:] = Omega_coords * scales + shifts
        self.radius_map = RadiusMap(G, edge_marker)

    def get_Omega_axis_plane(self, face: str) -> SubDomain:
        match face:
            case "left":
                return AxisPlane(0, np.min(self.Omega_bounds[:, 0]))
            case "right":
                return AxisPlane(0, np.max(self.Omega_bounds[:, 0]))
            case "bottom":
                return AxisPlane(1, np.min(self.Omega_bounds[:, 1]))
            case "top":
                return AxisPlane(1, np.max(self.Omega_bounds[:, 1]))
            case "front":
                return AxisPlane(2, np.min(self.Omega_bounds[:, 2]))
            case "back":
                return AxisPlane(2, np.max(self.Omega_bounds[:, 2]))
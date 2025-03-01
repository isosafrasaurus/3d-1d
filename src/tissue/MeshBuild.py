import numpy as np
import warnings
from typing import Optional, Tuple
from dolfin import UnitCubeMesh, MeshFunction, UserExpression, SubDomain, BoundingBoxTree
from graphnics import FenicsGraph
from rtree import index as rtree_index

class AxisPlane(SubDomain):
    def __init__(self, axis, coordinate: float):
        super().__init__()
        self.axis = axis
        self.coordinate = coordinate

    def inside(self, x, on_boundary: bool) -> bool:
        return on_boundary and near(x[self.axis], self.coordinate)

class RadiusMap(UserExpression):
    def __init__(self, G, Lambda):
        super().__init__()
        self.G = G
        self.tree = BoundingBoxTree()
        self.tree.build(Lambda)
    
    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        cell = self.tree.compute_first_entity_collision(p)
        edge_ix = self.G.mf[cell]
        edge = list(self.G.edges())[edge_ix]
        value[0] = self.G.edges()[edge]['radius']

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
        self.Lambda, _ = G.get_mesh(n = Lambda_num_nodes_exp)
        Lambda_coords = self.Lambda.coordinates()
        
        self.Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = self.Omega.coordinates()

        if Omega_bounds is None:
            min_lambda = np.min(Lambda_coords, axis=0)
            max_lambda = np.max(Lambda_coords, axis=0)
            scales = max_lambda - min_lambda + 2 * Lambda_padding
            shifts = min_lambda - Lambda_padding
            self.Omega_bounds = np.array([shifts.tolist(), (shifts + scales).tolist()])
        else:
            lower = np.minimum(Omega_bounds[0], Omega_bounds[1])
            upper = np.maximum(Omega_bounds[0], Omega_bounds[1])
            scales = upper - lower
            shifts = lower
            self.Omega_bounds = np.concatenate((lower, upper), axis=0)

        Omega_coords[:] = Omega_coords * scales + shifts
        self.radius_map = RadiusMap(G, self.Lambda)

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
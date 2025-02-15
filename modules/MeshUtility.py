from typing import Optional, List
from dolfin import UnitCubeMesh
from graphnics import *
from xii import *
import networkx as nx
import numpy as np

class MeshUtility:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 0.008,
        Lambda_num_nodes_exp: int = 5,
    ):
        G.make_mesh(n=Lambda_num_nodes_exp)
        G.make_submeshes()
        Lambda, Lambda_edge_marker = G.get_mesh(n=Lambda_num_nodes_exp)

        Lambda_coords = Lambda.coordinates()
        min_coords = np.min(Lambda_coords, axis=0)
        offset = np.where(min_coords < 0, -min_coords, 0)
        Lambda_coords[:] = Lambda_coords + offset

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
            
        Omega_coords[:,:] = Omega_coords * scales + shifts
        
        self.Omega_bounds = [shifts, shifts + scales]
        self.Omega = Omega
        self.Lambda = Lambda
        self.Lambda_edge_marker = Lambda_edge_marker
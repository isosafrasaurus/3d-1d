from typing import Optional, List
from dolfin import (
    MeshFunction, SubDomain, Measure, near, DOLFIN_EPS
)
from graphnics import FenicsGraph
from xii import *
import networkx as nx
import numpy as np
import importlib
import MeshCreator

class Face(SubDomain):
    pass

class XZeroPlane(Face):
    def inside(self, x, on_boundary):
        return on_boundary and np.abs(x[0] - 0.0) < 1

class MeasureMeshCreator(MeshCreator.MeshCreator):
    def __init__(
        self,
        G: FenicsGraph,
        Lambda_inlet: Optional[List[int]],
        Omega_sink: Face,
        Omega_bounds_dim: Optional[List[List[float]]],
        Omega_mesh_voxel_dim: List[int],
        Lambda_padding_min: float,
        Lambda_num_nodes_exp: int
    ):
    
        importlib.reload(MeshCreator)
        
        super().__init__(
            G,
            Omega_bounds_dim=Omega_bounds_dim,
            Omega_mesh_voxel_dim=Omega_mesh_voxel_dim,
            Lambda_padding_min=Lambda_padding_min,
            Lambda_num_nodes_exp=Lambda_num_nodes_exp
        )

        # Mark the sink boundary on Omega
        boundary_Omega = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        Omega_sink.mark(boundary_Omega, 1)

        # Create boundary markers for Lambda endpoints
        Lambda_boundary_markers = MeshFunction("size_t", self.Lambda, self.Lambda.topology().dim() - 1, 0)

        # If inlet points are specified, mark them as 1
        if Lambda_inlet is not None:
            for node_id in Lambda_inlet:
                pos = self.G_copy.nodes[node_id]["pos"]

                class InletEndpoint(SubDomain):
                    def __init__(self, point):
                        super().__init__()
                        self.point = point
                        print(str(point) + " MARKED!")

                    def inside(self, x, on_boundary):
                        return (
                            on_boundary
                            and not near(x[0], self.point[0])
                            and not near(x[1], self.point[1])
                            and not near(x[2], self.point[2])
                        )

                inlet_subdomain = InletEndpoint(pos)
                inlet_subdomain.mark(Lambda_boundary_markers, 1)

        # Define interior measures
        dxOmega = Measure("dx", domain=self.Omega)
        dxLambda = Measure("dx", domain=self.Lambda)

        # Define boundary measures for Omega
        dsOmega = Measure("ds", domain=self.Omega, subdomain_data=boundary_Omega)
        dsOmegaNeumann = dsOmega(0)  # Non-sink boundaries (Neumann)
        dsOmegaSink = dsOmega(1)     # Sink boundary

        # Define boundary measures for Lambda
        dsLambda = Measure("ds", domain=self.Lambda, subdomain_data=Lambda_boundary_markers)
        dsLambdaRobin = dsLambda(0)  # Endpoints not marked as inlet (Robin)
        dsLambdaInlet = dsLambda(1)    # Endpoints marked as inlet (Dirichlet)

        # Assign additional results as fields of the object
        self.boundary_Omega = boundary_Omega
        self.Lambda_boundary_markers = Lambda_boundary_markers
        self.dxOmega = dxOmega
        self.dxLambda = dxLambda
        self.dsOmegaNeumann = dsOmegaNeumann
        self.dsOmegaSink = dsOmegaSink
        self.dsLambdaRobin = dsLambdaRobin
        self.dsLambdaInlet = dsLambdaInlet
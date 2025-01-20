from typing import Optional, Dict, Any, List
from dolfin import (
    MeshFunction, SubDomain, Measure, near, DOLFIN_EPS
)
from graphnics import FenicsGraph
from xii import *
import networkx as nx
import numpy as np
import MeshCreator

class Face(SubDomain):
    pass

class XZeroPlane(Face):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class MeasureMeshCreator:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_sink: Face,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [32, 32, 32],
        Lambda_padding_min: float = 8,
        Lambda_num_nodes_exp: int = 8,
        Lambda_inlet: Optional[List[int]] = None
    ):
        self.G = G
        self.Omega_sink = Omega_sink
        self.inlet_points = Lambda_inlet
        self.Omega_bounds_dim = Omega_bounds_dim

    def create_mesh_and_measures(self) -> Dict[str, Any]:
        # Create the base meshes using the MeshCreator module function
        mesh_data = MeshCreator.create_mesh(self.G, Omega_bounds_dim=self.Omega_bounds_dim)
        Lambda = mesh_data["Lambda"]
        Omega = mesh_data["Omega"]
        Lambda_edge_marker = mesh_data["Lambda_edge_marker"]
        G_copy = mesh_data["G_copy"]

        # Mark the sink boundary on Omega
        boundary_Omega = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)
        self.Omega_sink.mark(boundary_Omega, 1)

        # Create boundary markers for Lambda endpoints, defaulting to 0
        Lambda_boundary_markers = MeshFunction("size_t", Lambda, Lambda.topology().dim() - 1, 0)

        # If inlet points are specified, mark them as 1
        if self.inlet_points is not None:
            for node_id in self.inlet_points:
                pos = G_copy.nodes[node_id]["pos"]

                class InletEndpoint(SubDomain):
                    def __init__(self, point):
                        super().__init__()
                        self.point = point

                    def inside(self, x, on_boundary):
                        return (
                            on_boundary
                            and near(x[0], self.point[0], DOLFIN_EPS)
                            and near(x[1], self.point[1], DOLFIN_EPS)
                            and near(x[2], self.point[2], DOLFIN_EPS)
                        )

                inlet_subdomain = InletEndpoint(pos)
                inlet_subdomain.mark(Lambda_boundary_markers, 1)

        # Define interior measures
        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)

        # Define boundary measures for Omega
        dsOmega = Measure("ds", domain=Omega, subdomain_data=boundary_Omega)
        dsOmegaNeumann = dsOmega(0)  # Non-sink boundaries (Neumann)
        dsOmegaSink = dsOmega(1)     # Sink boundary

        # Define boundary measures for Lambda
        dsLambda = Measure("ds", domain=Lambda, subdomain_data=Lambda_boundary_markers)
        dsLambdaNeumann = dsLambda(0)  # Endpoints not marked as inlet (Neumann)
        dsLambdaInlet = dsLambda(1)    # Endpoints marked as inlet (Dirichlet)

        mesh_data.update({
            "boundary_Omega": boundary_Omega,
            "Lambda_boundary_markers": Lambda_boundary_markers,
            "dxOmega": dxOmega,
            "dxLambda": dxLambda,
            "dsOmegaNeumann": dsOmegaNeumann,
            "dsOmegaSink": dsOmegaSink,
            "dsLambdaNeumann": dsLambdaNeumann,
            "dsLambdaInlet": dsLambdaInlet,
            "edge_marker": Lambda_edge_marker,
        })

        return mesh_data

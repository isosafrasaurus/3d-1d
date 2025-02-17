from typing import Optional, List
from dolfin import MeshFunction, SubDomain, Measure, near, DOLFIN_EPS, Mesh
from graphnics import FenicsGraph
from xii import *
import numpy as np
import importlib
import LiverMesh
import os

importlib.reload(LiverMesh)

class Face(SubDomain):
    pass

class XAxisPlane(Face):
    def __init__(self, lock):
        super().__init__()
        self.lock = lock
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.lock)

class InletEndpoint(SubDomain):
    def __init__(self, point):
        super().__init__()
        self.point = point
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and near(x[0], self.point[0])
            and near(x[1], self.point[1])
            and near(x[2], self.point[2])
        )

class LiverMeshMeasure(LiverMesh.LiverMesh):
    def __init__(
        self,
        G: FenicsGraph,
        Lambda_inlet: List[int],
        Omega_sink: SubDomain,
        **kwargs
    ):
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(G, **filtered_kwargs)

        if Omega_sink == None:
            Omega_sink = XAxisPlane(self.Omega_bounds[0][0])

        # Markers for sink on Omega
        boundary_Omega = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        Omega_sink.mark(boundary_Omega, 1)

        # Markers for endpoints on Lambda
        boundary_Lambda = MeshFunction("size_t", self.Lambda, self.Lambda.topology().dim() - 1, 0)
        if Lambda_inlet is not None:
            lambda_coordinates = self.Lambda.coordinates()
            for node_id in Lambda_inlet:
                if node_id < 0 or node_id >= len(lambda_coordinates):
                    raise ValueError(f"Lambda_inlet node_id {node_id} is out of bounds for the Lambda mesh.")
                pos = lambda_coordinates[node_id]
                inlet_subdomain = InletEndpoint(pos)
                inlet_subdomain.mark(boundary_Lambda, 1)

        self.dxOmega = Measure("dx", domain=self.Omega)
        self.dxLambda = Measure("dx", domain=self.Lambda)
        self.dsOmega = Measure("ds", domain=self.Omega, subdomain_data=boundary_Omega)
        self.dsOmegaNeumann = self.dsOmega(0)  # Non-sink boundaries (Neumann)
        self.dsOmegaSink = self.dsOmega(1)     # Sink boundary (Robin)
        self.dsLambda = Measure("ds", domain=self.Lambda, subdomain_data=boundary_Lambda)
        self.dsLambdaRobin = self.dsLambda(0)    # Endpoints not marked as inlet (Robin)
        self.dsLambdaInlet = self.dsLambda(1)    # Endpoints marked as inlet (Dirichlet)
        self.boundary_Omega = boundary_Omega
        self.boundary_Lambda = boundary_Lambda
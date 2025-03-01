import numpy as np
import warnings
from .MeshBuild import MeshBuild
from dolfin import MeshFunction, SubDomain, Measure, near, DOLFIN_EPS, Mesh
from graphnics import FenicsGraph
from typing import Optional, List

class BoundaryPoint(SubDomain):
    def __init__(self, coordinate):
        super().__init__()
        self.coordinate = coordinate

    def inside(self, x, on_boundary: bool) -> bool:
        return (
            on_boundary
            and near(x[0], self.coordinate[0])
            and near(x[1], self.coordinate[1])
            and near(x[2], self.coordinate[2])
        )

class MeasureBuild():
    def __init__(
        self,
        mesh: MeshBuild,
        Lambda_inlet: Optional[List[int]] = None,
        Omega_sink: Optional[SubDomain] = None
    ):
        self.mesh = mesh
        self.Lambda_inlet = Lambda_inlet
        self.Omega_sink = Omega_sink
        
        boundary_Omega = MeshFunction("size_t", mesh.Omega, mesh.Omega.topology().dim() - 1, 0)
        boundary_Lambda = MeshFunction("size_t", mesh.Lambda, mesh.Lambda.topology().dim() - 1, 0)
        
        if Omega_sink is not None:
            Omega_sink.mark(boundary_Omega, 1)
            self.dsOmega = Measure("ds", domain=mesh.Omega, subdomain_data=boundary_Omega)
            self.dsOmegaNeumann = self.dsOmega(0)
            self.dsOmegaSink = self.dsOmega(1)
        else:
            warnings.warn("No Lambda inlets provided. Defaulting to Neumann conditions.")
            self.dsOmega = Measure("ds", domain=mesh.Omega)
        
        if Lambda_inlet is not None:
            lambda_coordinates = mesh.Lambda.coordinates()
            for node_id in Lambda_inlet:
                if not (0 <= node_id < len(lambda_coordinates)):
                    raise ValueError(f"Lambda_inlet node_id {node_id} is out of bounds for the Lambda mesh.")
                coordinate = lambda_coordinates[node_id]
                inlet_subdomain = BoundaryPoint(coordinate)
                inlet_subdomain.mark(boundary_Lambda, 1)
            self.dsLambda = Measure("ds", domain=mesh.Lambda, subdomain_data=boundary_Lambda)
            self.dsLambdaRobin = self.dsLambda(0)
            self.dsLambdaInlet = self.dsLambda(1)
        else:
            warnings.warn("No Lambda inlets provided.")
            self.dsLambda = Measure("ds", domain=mesh.Lambda)

        self.dxOmega = Measure("dx", domain=mesh.Omega)
        self.dxLambda = Measure("dx", domain=mesh.Lambda)
        self.boundary_Omega = boundary_Omega
        self.boundary_Lambda = boundary_Lambda
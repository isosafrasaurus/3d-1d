# MeasureMeshCreator.py

from typing import Optional, List
from dolfin import (
    MeshFunction, SubDomain, Measure, near, DOLFIN_EPS, Mesh
)
from graphnics import FenicsGraph
from xii import *
import numpy as np
import importlib
import MeshCreator
import os

class Face(SubDomain):
    pass

class XZeroPlane(Face):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class MeasureMeshCreator(MeshCreator.MeshCreator):
    def __init__(
        self,
        G: FenicsGraph,
        Lambda_inlet: Optional[List[int]],
        Omega_sink: SubDomain,
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
            # Access mesh coordinates directly
            lambda_coordinates = self.Lambda.coordinates()

            for node_id in Lambda_inlet:
                # Ensure node_id is within the valid range
                if node_id < 0 or node_id >= len(lambda_coordinates):
                    raise ValueError(f"Lambda_inlet node_id {node_id} is out of bounds for the Lambda mesh.")

                pos = lambda_coordinates[node_id]

                class InletEndpoint(SubDomain):
                    def __init__(self, point, tol=1e-8):
                        super().__init__()
                        self.point = point
                        self.tol = tol
                        print(f"Inlet endpoint at {self.point} marked!")

                    def inside(self, x, on_boundary):
                        # Mark only those boundary points that are near self.point
                        return (
                            on_boundary
                            and near(x[0], self.point[0], self.tol)
                            and near(x[1], self.point[1], self.tol)
                            and near(x[2], self.point[2], self.tol)
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
        dsLambdaRobin = dsLambda(0)    # Endpoints not marked as inlet (Robin)
        dsLambdaInlet = dsLambda(1)    # Endpoints marked as inlet (Dirichlet)

        # Assign additional results as fields of the object
        self.boundary_Omega = boundary_Omega
        self.Lambda_boundary_markers = Lambda_boundary_markers
        self.dxOmega = dxOmega
        self.dxLambda = dxLambda
        self.dsOmega = dsOmega
        self.dsLambda = dsLambda
        self.dsOmegaNeumann = dsOmegaNeumann
        self.dsOmegaSink = dsOmegaSink
        self.dsLambdaRobin = dsLambdaRobin
        self.dsLambdaInlet = dsLambdaInlet

    def check_inlet_boundary(self):
        """
        Checks whether the 1D inlet boundary (dsLambdaInlet) is applied.
        Prints the number of endpoints marked as inlet and a confirmation message.
        """
        inlet_markers = self.Lambda_boundary_markers

        try:
            # For older FEniCS versions
            inlet_array = inlet_markers.array()
        except AttributeError:
            try:
                # For newer FEniCS versions
                inlet_array = inlet_markers.values()
            except AttributeError:
                # As a fallback, use .vector().get_local()
                inlet_array = inlet_markers.vector().get_local()

        num_inlet_facets = np.sum(inlet_array == 1)

        if num_inlet_facets > 0:
            print(f"[CHECK] 1D Inlet boundary is applied on {num_inlet_facets} facet(s).")
        else:
            print("[WARNING] 1D Inlet boundary is NOT applied on any facets.")

    def export_boundary_markers(self, directory_path: str):
        """
        Exports the boundary markers to VTK files for visualization.
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # Export Omega boundaries
        omega_vtk = os.path.join(directory_path, "boundary_Omega.pvd")
        File(omega_vtk) << self.boundary_Omega
        
        # Export Lambda boundaries
        lambda_vtk = os.path.join(directory_path, "boundary_Lambda.pvd")
        File(lambda_vtk) << self.Lambda_boundary_markers

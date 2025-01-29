from dolfin import *
from graphnics import *
from xii import *
from typing import Optional, List, Any
import numpy as np
import MeasureMeshCreator
import VTKExporter
import importlib
import RadiusFunction
import os

class FEMSink:
    def __init__(
        self,
        G: "FenicsGraph",
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        gamma_v: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float,
        Lambda_inlet: List[int],
        Omega_sink: SubDomain = MeasureMeshCreator.XZeroPlane(),
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [32, 32, 32],
        Lambda_padding_min: float = 8,
        Lambda_num_nodes_exp: int = 8
    ):
        importlib.reload(MeasureMeshCreator)
    
        measure_creator = MeasureMeshCreator.MeasureMeshCreator(
            G,
            Lambda_inlet,
            Omega_sink,
            Omega_bounds_dim=Omega_bounds_dim,
            Omega_mesh_voxel_dim=Omega_mesh_voxel_dim,
            Lambda_padding_min=Lambda_padding_min,
            Lambda_num_nodes_exp=Lambda_num_nodes_exp
        )
        
        self.Omega = measure_creator.Omega
        self.Lambda = measure_creator.Lambda
        self.dsOmegaSink = measure_creator.dsOmegaSink
        self.dsOmegaNeumann = measure_creator.dsOmegaNeumann
        self.dsLambdaInlet = measure_creator.dsLambdaInlet
        self.dsLambdaRobin = measure_creator.dsLambdaRobin
        self.dxOmega = measure_creator.dxOmega
        self.dxLambda = measure_creator.dxLambda

        # **Access the MeshFunction for Omega boundaries**
        self.boundary_Omega = measure_creator.boundary_Omega  # Added line

        self.Lambda_boundary_markers = measure_creator.Lambda_boundary_markers  # Ensure this is accessible

        self.mu = mu
        self.k_t = k_t
        self.k_v = k_v
        self.gamma = gamma
        self.gamma_R = gamma_R
        self.gamma_a = gamma_a
        self.p_cvp = p_cvp
        self.P_in = P_in

        # Define function spaces and trial/test functions
        V3 = FunctionSpace(self.Omega, "CG", 1)
        V1 = FunctionSpace(self.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        self.radius_map = RadiusFunction.RadiusFunction(G, measure_creator.Lambda_edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)

        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        # Assemble system matrices
        a00 = (
            Constant(self.k_t / self.mu) * inner(grad(u3), grad(v3)) * self.dxOmega
            + Constant(self.gamma) * u3_avg * v3_avg * D_perimeter * self.dxLambda
            + Constant(self.gamma_R) * u3 * v3 * self.dsOmegaSink
        )
        a01 = -Constant(self.gamma) * u1 * v3_avg * D_perimeter * self.dxLambda
        a10 = -Constant(self.gamma) * u3_avg * v1 * D_perimeter * self.dxLambda
        a11 = (
            Constant(self.k_v / self.mu) * inner(grad(u1), grad(v1)) * D_area * self.dxLambda
            + Constant(self.gamma) * u1 * v1 * D_perimeter * self.dxLambda
            - Constant(self.gamma_a / self.mu) * u1 * v1 * self.dsLambdaRobin
        )
        a = [[a00, a01],
             [a10, a11]]

        L0 = -Constant(self.gamma_R) * Constant(self.p_cvp) * v3 * self.dsOmegaSink
        L1 = -Constant(self.gamma_a / self.mu) * Constant(self.p_cvp) * v1 * self.dsLambdaRobin
        L = [L0, L1]

        # Boundary conditions: apply Dirichlet BC on 1D inlet where marker = 1
        inlet_bc = DirichletBC(V1, Constant(self.P_in), self.Lambda_boundary_markers, 1)
        inlet_bcs = [inlet_bc] if len(inlet_bc.get_boundary_values()) > 0 else []
        W_bcs = [[], inlet_bcs]

        # Apply boundary conditions if any
        A, b = map(ii_assemble, (a, L))
        if any(W_bcs[0]) or any(W_bcs[1]):
            print("Applied BC! Non-empty list")
            A, b = apply_bc(A, b, W_bcs)
        else:
            print("WARNING! No Dirichlet BCs applied!")
        A, b = map(ii_convert, (A, b))

        # Solve the system
        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)

        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure", "1D Pressure Distribution")

    def save_vtk(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)

        out_1d = os.path.join(directory_path, "pressure1d.vtk")
        out_3d = os.path.join(directory_path, "pressure3d.pvd")

        VTKExporter.fenics_to_vtk(
            self.Lambda,
            out_1d,
            self.radius_map,
            uh1d=self.uh1d
        )
        File(out_3d) << self.uh3d

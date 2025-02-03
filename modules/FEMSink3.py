from dolfin import *
from graphnics import *
from xii import *
from typing import List
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
        **kwargs
    ):
        importlib.reload(MeasureMeshCreator)

        mm_kwargs = {k: v for k, v in kwargs.items() if v is not None}    
        measure_creator = MeasureMeshCreator.MeasureMeshCreator(
            G,
            Lambda_inlet,
            Omega_sink,
            **mm_kwargs
        )
        
        self.Omega = measure_creator.Omega
        self.Lambda = measure_creator.Lambda
        self.dsOmegaSink = measure_creator.dsOmegaSink
        self.dsOmegaNeumann = measure_creator.dsOmegaNeumann
        self.dsLambdaInlet = measure_creator.dsLambdaInlet
        # **NOTE:** We will no longer use dsLambdaRobin directly for point coupling.
        self.dxOmega = measure_creator.dxOmega
        self.dxLambda = measure_creator.dxLambda

        # Access boundary markers if needed
        self.boundary_Omega = measure_creator.boundary_Omega
        self.Lambda_boundary_markers = measure_creator.Lambda_boundary_markers

        self.mu = mu
        self.k_t = k_t
        self.k_v = k_v
        self.gamma = gamma
        self.gamma_R = gamma_R
        self.gamma_a = gamma_a
        self.p_cvp = p_cvp
        self.P_in = P_in

        # Define function spaces
        V3 = FunctionSpace(self.Omega, "CG", 1)
        V1 = FunctionSpace(self.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        # Create the lateral average operators for coupling
        self.radius_map = RadiusFunction.RadiusFunction(G, measure_creator.Lambda_edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        # Vessel cross-sectional area and perimeter
        D_area = np.pi * self.radius_map**2        # |Theta|
        D_perimeter = 2.0 * np.pi * self.radius_map   # |∂Theta|

        # Assemble system matrices
        # (Notice: We have removed any terms using dsLambdaRobin, so that no codim >1 measure appears.)
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
            # The vessel outlet Robin condition is already built into the 1D weak form (via boundary integration)
            - Constant(self.gamma_a / self.mu) * u1 * v1 * measure_creator.dsLambdaRobin
        )
        a = [[a00, a01],
             [a10, a11]]

        L0 = -Constant(self.gamma_R) * Constant(self.p_cvp) * v3 * self.dsOmegaSink
        L1 = -Constant(self.gamma_a / self.mu) * Constant(self.p_cvp) * v1 * measure_creator.dsLambdaRobin
        L = [L0, L1]

        # Apply Dirichlet BC on the 1D inlet (marker 1)
        inlet_bc = DirichletBC(V1, Constant(self.P_in), self.Lambda_boundary_markers, 1)
        inlet_bcs = [inlet_bc] if len(inlet_bc.get_boundary_values()) > 0 else []
        W_bcs = [[], inlet_bcs]

        A, b = map(ii_assemble, (a, L))
        if any(W_bcs[0]) or any(W_bcs[1]):
            A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        outlet_points = measure_creator.get_outlet_points()  # (Assume this function returns the physical coordinates)
        for pt in outlet_points:
            # Here we assume P(A_i) is known or approximated.
            # For a linear problem you might compute it from the current 1D solution.
            # For demonstration we use P_in as a placeholder.
            P_out = self.P_in  # or get from an extrapolation of the 1D solution
            s = (self.gamma_a / self.mu) * (D_area) * (P_out - self.p_cvp)
            ps = PointSource(V3, Point(pt), s)
            ps.apply(b[0])  # Apply only to the 3D component of the RHS
            
        # Solve the coupled system
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

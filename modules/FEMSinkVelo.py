from dolfin import *
from graphnics import *
from xii import *
from rtree import index as rtree_index
from typing import Optional, List, Any
import numpy as np
import os
import random

from MeasureMeshCreator import MeasureMeshCreator, XZeroPlane
import VTKExporter
from RadiusFunction import RadiusFunction

class FEMSinkVelo:
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
        Lambda_inlet: Optional[List[int]] = None,
        Omega_sink: SubDomain = XZeroPlane(),
        Omega_bounds_dim: Optional[List[List[int]]] = None
    ):
    
        measure_creator = MeasureMeshCreator(G, Omega_sink, Lambda_inlet=Lambda_inlet)
        
        self.Omega = measure_creator.Omega
        self.Lambda = measure_creator.Lambda
        self.dsOmegaSink = measure_creator.dsOmegaSink
        self.dsOmegaNeumann = measure_creator.dsOmegaNeumann
        self.dsLambdaInlet = measure_creator.dsLambdaInlet
        self.dsLambdaNeumann = measure_creator.dsLambdaNeumann
        self.dxOmega = measure_creator.dxOmega
        self.dxLambda = measure_creator.dxLambda

        self.boundary_Omega = measure_creator.boundary_Omega
        self.Lambda_boundary_markers = measure_creator.Lambda_boundary_markers
        self.edge_marker = measure_creator.edge_marker
        self.G_copy = measure_creator.G_copy

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

        self.radius_map = RadiusFunction(G, self.edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)

        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        # Assemble system matrices
        a00 = (
            Constant(self.k_t/self.mu) * inner(grad(u3), grad(v3)) * self.dxOmega
            + Constant(self.gamma_R) * u3 * v3 * self.dsOmegaSink
            - Constant(self.gamma) * u3_avg * v3_avg * D_perimeter * self.dxLambda
        )
        a01 = Constant(self.gamma) * u1 * v3_avg * D_perimeter * self.dxLambda
        a10 = -Constant(self.gamma) * u3_avg * v1 * D_perimeter * self.dxLambda
        a11 = (
            Constant(self.k_v/self.mu) * inner(grad(u1), grad(v1)) * D_area * self.dxLambda
            + Constant(self.gamma) * u1 * v1 * D_perimeter * self.dxLambda
            + Constant(self.gamma_a/self.mu) * u1 * v1 * self.dsLambdaNeumann
        )
        a = [[a00, a01],
             [a10, a11]]

        L0 = - Constant(self.gamma_R) * Constant(self.p_cvp) * v3 * self.dsOmegaSink
        L1 = - Constant(self.gamma_a/self.mu) * Constant(self.p_cvp) * v1 * self.dsLambdaNeumann
        L = [L0, L1]

        # Boundary conditions: apply Dirichlet BC on 1D inlet where marker = 1
        inlet_bc = DirichletBC(W[1], Constant(self.P_in), self.Lambda_boundary_markers, 1)
        inlet_bcs = [inlet_bc] if len(inlet_bc.get_boundary_values()) > 0 else []
        W_bcs = [[], inlet_bcs]

        A, b = map(ii_assemble, (a, L))
        if any(W_bcs[0]) or any(W_bcs[1]):  # Only apply BC if non-empty lists exist
            print("Applied BC! Non-empty list")
            A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)

        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure", "1D Pressure Distribution")

        # Compute velocity in 3D: u_3 = -(k_t/mu)*grad(uh3d)
        V_vector = VectorFunctionSpace(self.Omega, "CG", 1)
        u_vel = TrialFunction(V_vector)
        v_vel = TestFunction(V_vector)

        a_vel = dot(u_vel, v_vel)*dx
        L_vel = -Constant(self.k_t/self.mu)*dot(grad(self.uh3d), v_vel)*self.dxOmega

        A_vel = assemble(a_vel)
        b_vel = assemble(L_vel)
        solver_vel = LUSolver(A_vel, "mumps")
        velocity = Function(V_vector, name="Velocity Field")
        solver_vel.solve(velocity.vector(), b_vel)
        velocity.rename("Velocity", "Velocity Field")
        self.velocity = velocity

    def calculate_3d_outflow(self) -> float:
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*self.dsOmegaSink)

    def calculate_1d_inflow(self) -> float:
        mesh1d = self.Lambda
        v0 = Vertex(mesh1d, 0)
        edges = list(v0.entities(1))
        if not edges:
            return 0.0

        e = Edge(mesh1d, edges[0])
        vs = list(e.entities(0))
        neighbor_v = vs[0] if vs[0] != v0.index() else vs[1]
        v1 = Vertex(mesh1d, neighbor_v)

        coords0 = np.array(v0.point().array())
        coords1 = np.array(v1.point().array())
        dist = np.linalg.norm(coords1 - coords0)
        if dist < 1e-14:
            return 0.0

        p0 = self.uh1d(v0.point())
        p1 = self.uh1d(v1.point())
        dP_ds = (p1 - p0)/dist

        r0 = float(self.radius_map(v0.point()))
        A0 = np.pi*(r0**2)
        return -A0*(self.k_v/self.mu)*dP_ds

    def calculate_1d_outflow(self) -> float:
        boundary_markers = self.dsLambdaNeumann.subdomain_data()
        mesh1d = self.Lambda
        outflow_total = 0.0

        for f in facets(mesh1d):
            if boundary_markers[f.index()] == 1:
                vs = list(f.entities(0))
                for vid in vs:
                    v = Vertex(mesh1d, vid)
                    pval = self.uh1d(v.point())
                    flux_i = (self.gamma_a/self.mu)*(pval - self.p_cvp)
                    outflow_total += flux_i

        return outflow_total

    def calculate_total_outflow(self) -> float:
        return self.calculate_3d_outflow() + self.calculate_1d_outflow()

    def calculate_total_flow_all_boundaries(self) -> float:
        ds_all = Measure("ds", domain=self.Omega)
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*ds_all)

    def save_vtk(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)

        out_1d = os.path.join(directory_path, "pressure1d.vtk")
        out_3d = os.path.join(directory_path, "pressure3d.pvd")
        out_vel = os.path.join(directory_path, "velocity3d.pvd")

        VTKExporter.fenics_to_vtk(
            self.Lambda,
            out_1d,
            self.radius_map,
            uh1d=self.uh1d
        )
        File(out_3d) << self.uh3d
        File(out_vel) << self.velocity

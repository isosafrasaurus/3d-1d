import os, fem, tissue, visualize
import numpy as np
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, File, PETScKrylovSolver
from graphnics import ii_assemble, apply_bc, ii_convert, ii_Function
from xii import Circle, Average

class SinkNoReinject:
    def __init__(
        self,
        domain: tissue.MeasureBuild,
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        gamma_v: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float
    ):
        # Set attributes
        for name, value in zip(
            ["gamma", "gamma_a", "gamma_R", "gamma_v", "mu", "k_t", "k_v", "P_in", "p_cvp"],
            [gamma, gamma_a, gamma_R, gamma_v, mu, k_t, k_v, P_in, p_cvp]
        ):
            setattr(self, name, value)
        for attr in ["Omega", "Lambda", "radius_map"]:
            setattr(self, attr, getattr(domain.mesh, attr))
        for attr in ["dxOmega", "dxLambda", "dsOmegaNeumann", "dsOmegaSink","dsLambdaRobin", "dsLambdaInlet", "boundary_Lambda"]:
            setattr(self, attr, getattr(domain, attr))

        # Function spaces
        V3 = FunctionSpace(self.Omega, "CG", 1)
        V1 = FunctionSpace(self.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        # Averaging operators
        cylinder = Circle(radius=self.radius_map, degree=5)
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)
        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        # Block matrix terms
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

        # Inlet Dirichlet conditions
        inlet_bc = DirichletBC(V1, Constant(self.P_in), self.boundary_Lambda, 1)
        inlet_bcs = [inlet_bc] if inlet_bc.get_boundary_values() else []
        W_bcs = [[], inlet_bcs]

        # Solve
        A, b = map(ii_assemble, (a, L))
        if any(W_bcs[0]) or any(W_bcs[1]):
            A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))
        wh = ii_Function(W)
        solver = PETScKrylovSolver("cg", "hypre_amg")
        solver.set_operator(A)
        solver.parameters["relative_tolerance"] = 1e-8
        solver.parameters["maximum_iterations"] = int(1e6)
        solver.solve(wh.vector(), b)

        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure (Pa)", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure (Pa)", "1D Pressure Distribution")

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        # Save Lambda solution
        visualize.save_Lambda(
            save_path = f"{directory}/pressure1d.vtk",
            Lambda = self.Lambda,
            radius_map = self.radius_map,
            uh1d = self.uh1d
        )
        # Save Omega solution
        File(f"{directory}/pressure3d.pvd") << self.uh3d
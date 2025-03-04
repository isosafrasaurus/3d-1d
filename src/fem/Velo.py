import os, tissue
import numpy as np
from dolfin import *
from .Sink import Sink

class Velo(Sink):
    def __init__(
        self,
        domain: tissue.MeasureBuild,
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float
    ):
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        V_vec = VectorFunctionSpace(self.Omega, "CG", 1)
        self.velocity = project(Constant(- self.k_t / self.mu) * grad(self.uh3d), V_vec)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaSink)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmega)

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        super().save_vtk(directory)
        self.velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        velocity_file = File(os.path.join(directory, "velocity3d.pvd"))
        velocity_file << self.velocity
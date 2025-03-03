import tissue
import numpy as np
from dolfin import *
from typing import List
from .Velo import Velo

class CubeSubBoundary(SubDomain):
    def __init__(self, lower, upper):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[0], self.lower[0]) or near(x[0], self.upper[0]) or
            near(x[1], self.lower[1]) or near(x[1], self.upper[1]) or
            near(x[2], self.lower[2]) or near(x[2], self.upper[2])
        )

class SubCubes(Velo):
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
        p_cvp: float,
        lower_cube_bounds: List[int],
        upper_cube_bounds: List[int]
    ):
        super().__init__(domain, gamma, gamma_a, gamma_R, gamma_v, mu, k_t, k_v, P_in, p_cvp)
        
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
            
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        
        self.lower_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.lower_boundaries.set_all(0)
        self.upper_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.upper_boundaries.set_all(0)
        
        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        self.ds_lower = Measure("ds", domain=self.Omega, subdomain_data=self.lower_boundaries)
        self.ds_upper = Measure("ds", domain=self.Omega, subdomain_data=self.upper_boundaries)
    
    def compute_lower_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_lower = assemble(dot(self.velocity, n) * self.ds_lower(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_upper = assemble(dot(self.velocity, n) * self.ds_upper(1))
        return flux_upper

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        flux_lower_in = assemble(conditional(dot(self.velocity, n) < 0,
                                             dot(self.velocity, n), 0) * self.ds_lower(1))
        return flux_lower_in

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        flux_lower_out = assemble(conditional(dot(self.velocity, n) > 0,
                                              dot(self.velocity, n), 0) * self.ds_lower(1))
        return flux_lower_out

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        flux_upper_in = assemble(conditional(dot(self.velocity, n) < 0,
                                             dot(self.velocity, n), 0) * self.ds_upper(1))
        return flux_upper_in

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        flux_upper_out = assemble(conditional(dot(self.velocity, n) > 0,
                                              dot(self.velocity, n), 0) * self.ds_upper(1))
        return flux_upper_out
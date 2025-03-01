import numpy as np
import math
import FEMSinkVelo
import importlib

from dolfin import *
from typing import List, Optional
from MeasureBuild import MeasureBuild

class FEMSinkCubeFlux(FEMSinkVelo.FEMSinkVelo):
    def __init__(
        self,
        mc: MeasureCreator,
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
        Omega_sink: SubDomain,
        lower_cube_bounds: Optional[List[List[float]]],
        upper_cube_bounds: Optional[List[List[float]]],
        **kwargs
    ):
        super().__init__(mc, gamma, gamma_a, gamma_R, gamma_v, mu, k_t, k_v, P_in, p_cvp, Lambda_inlet, Omega_sink, **kwargs)
        
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
        
        class CubeSubBoundary(SubDomain):
            def __init__(self, lower: List[float], upper: List[float], tol=1e-6):
                super().__init__()
                self.lower = lower
                self.upper = upper
                self.tol = tol

            def inside(self, x, on_boundary):
                return on_boundary and (
                    x[0] >= self.lower[0] - self.tol and x[0] <= self.upper[0] + self.tol and
                    x[1] >= self.lower[1] - self.tol and x[1] <= self.upper[1] + self.tol and
                    x[2] >= self.lower[2] - self.tol and x[2] <= self.upper[2] + self.tol
                )
        
        # Instantiate the two sub-boundaries for the cubes.
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
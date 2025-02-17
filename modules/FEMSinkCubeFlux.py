import numpy as np
import math
import FEMSinkVelo
import importlib

from dolfin import *
from typing import List, Optional
from MeasureCreator import MeasureCreator

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
        lower_cube_bounds: Optional[List[List[float]]] = None,
        upper_cube_bounds: Optional[List[List[float]]] = None,
        **kwargs
    ):
        super().__init__(mc, gamma, gamma_a, gamma_R, gamma_v, mu, k_t, k_v, P_in, p_cvp, Lambda_inlet, Omega_sink, **kwargs)
        
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        eps = 0.5 * min(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Use provided bounds if available, otherwise auto-generate them.
        # The expected format is [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        if lower_cube_bounds is None:
            lower_cube_bounds = [[x_min, y_min, z_min], [x_min + eps, y_min + eps, z_min + eps]]
        if upper_cube_bounds is None:
            upper_cube_bounds = [[x_max - eps, y_max - eps, z_max - eps], [x_max, y_max, z_max]]
            
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        
        # Instead of one MeshFunction for both, create two separate ones.
        self.lower_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.lower_boundaries.set_all(0)
        self.upper_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.upper_boundaries.set_all(0)
        
        # Define a local SubDomain for a cube sub-boundary.
        class CubeSubBoundary(SubDomain):
            def __init__(self, lower: List[float], upper: List[float], tol=1e-6):
                super().__init__()
                self.lower = lower
                self.upper = upper
                self.tol = tol

            def inside(self, x, on_boundary):
                return on_boundary and not (
                    x[0] >= self.lower[0] - self.tol and x[0] <= self.upper[0] + self.tol or
                    x[1] >= self.lower[1] - self.tol and x[1] <= self.upper[1] + self.tol or
                    x[2] >= self.lower[2] - self.tol and x[2] <= self.upper[2] + self.tol
                )
        
        # Instantiate the two sub-boundaries for the cubes.
        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        
        # Mark the facets for each cube on their respective MeshFunctions.
        # Here we use marker 1 for both, since they are separate.
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        
        # Create Measures (ds) that use our custom subdomain markers.
        self.ds_lower = Measure("ds", domain=self.Omega, subdomain_data=self.lower_boundaries)
        self.ds_upper = Measure("ds", domain=self.Omega, subdomain_data=self.upper_boundaries)
    
    def compute_lower_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_lower = assemble(abs(dot(self.velocity, n)) * self.ds_lower(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_upper = assemble(abs(dot(self.velocity, n)) * self.ds_upper(1))
        return flux_upper

    def save_vtk(self, directory_path: str):
        import os
        os.makedirs(directory_path, exist_ok=True)
        super().save_vtk(directory_path)
        # Optionally save both markers.
        xdmf_file_lower = XDMFFile(os.path.join(directory_path, "lower_cube_boundaries.xdmf"))
        xdmf_file_lower.write(self.lower_boundaries)
        xdmf_file_upper = XDMFFile(os.path.join(directory_path, "upper_cube_boundaries.xdmf"))
        xdmf_file_upper.write(self.upper_boundaries)
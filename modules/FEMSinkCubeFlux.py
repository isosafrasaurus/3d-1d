import numpy as np
import math
from dolfin import *
import FEMSinkVelo
import importlib
from typing import List, Optional

class FEMSinkCubeFlux(FEMSinkVelo.FEMSinkVelo):
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
        Omega_sink: SubDomain = None,
        lower_cube_bounds: Optional[List[List[float]]] = None,
        upper_cube_bounds: Optional[List[List[float]]] = None,
        **kwargs
    ):
        super().__init__(G, gamma, gamma_a, gamma_R, gamma_v, mu, k_t, k_v, P_in, p_cvp, Lambda_inlet, Omega_sink, **kwargs)
        
        # Get the domain coordinates.
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        # Choose a cube side length: here, 50% of the smallest domain dimension
        eps = 0.5 * min(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Use provided bounds if available, otherwise auto-generate them.
        # The expected format is [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        if lower_cube_bounds is None:
            lower_cube_bounds = [[x_min, y_min, z_min], [x_min + eps, y_min + eps, z_min + eps]]
        if upper_cube_bounds is None:
            upper_cube_bounds = [[x_max - eps, y_max - eps, z_max - eps], [x_max, y_max, z_max]]
            
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        
        # Create a MeshFunction for the facets (boundary faces) of Omega.
        # We use the topological dimension of facets, i.e., dim-1.
        self.cube_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.cube_boundaries.set_all(0)
        
        # Define a local SubDomain for a cube sub-boundary.
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
        
        # Mark the facets that lie in each cube.
        # We use marker 1 for the lower cube and marker 2 for the upper cube.
        self.lower_cube.mark(self.cube_boundaries, 1)
        self.upper_cube.mark(self.cube_boundaries, 2)
        
        # Create a Measure (ds) that uses our custom subdomain markers.
        self.ds_cube = Measure("ds", domain=self.Omega, subdomain_data=self.cube_boundaries)
    
    def compute_lower_cube_flux(self):
        """
        Computes the flux through the lower cube sub-boundary.
        
        Returns:
            flux_lower (float): The integrated flux over the lower cube.
        """
        n = FacetNormal(self.Omega)
        flux_lower = assemble(dot(self.velocity, n) * self.ds_cube(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        """
        Computes the flux through the upper cube sub-boundary.
        
        Returns:
            flux_upper (float): The integrated flux over the upper cube.
        """
        n = FacetNormal(self.Omega)
        flux_upper = assemble(dot(self.velocity, n) * self.ds_cube(2))
        return flux_upper

    def save_vtk(self, directory_path: str):
        """
        Saves the 3D and 1D pressure fields, the velocity field, and also the cube
        boundary markers to the specified directory for post-processing.
        """
        import os
        os.makedirs(directory_path, exist_ok=True)
        # Save the pressure and velocity fields using the parent's method.
        super().save_vtk(directory_path)
        # Save the cube boundaries (for example, as an XDMF file) so that you can visualize them.
        xdmf_file = XDMFFile(os.path.join(directory_path, "cube_boundaries.xdmf"))
        xdmf_file.write(self.cube_boundaries)

import numpy as np
import math
from dolfin import *
import FEMSink2
import importlib

importlib.reload(FEMSink2)

# Assuming FEMSinkVelo is available; adjust the import path if needed.
from FEMSinkVelo import FEMSinkVelo  

class FEMSinkCubeFlux(FEMSinkVelo):
    def __init__(self, *args, **kwargs):
        # Initialize parent class (this sets up the problem, solves it, and computes velocity)
        super().__init__(*args, **kwargs)
        
        # Retrieve all vertex coordinates of Omega to define the bounding box
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        # Choose a cube side length: here, 10% of the smallest domain dimension
        eps = 0.5 * min(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Define the lower cube sub-boundary (near the (x_min, y_min, z_min) corner)
        self.lower_cube_bounds = (x_min, x_min + eps, y_min, y_min + eps, z_min, z_min + eps)
        # Define the upper cube sub-boundary (near the (x_max, y_max, z_max) corner)
        self.upper_cube_bounds = (x_max - eps, x_max, y_max - eps, y_max, z_max - eps, z_max)
        
        # Create a MeshFunction for the facets (boundary faces) of Omega.
        # We use the topological dimension of facets, i.e., dim-1.
        self.cube_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.cube_boundaries.set_all(0)
        
        # Define a local SubDomain for a cube sub-boundary.
        class CubeSubBoundary(SubDomain):
            def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, tol=1e-6):
                super().__init__()
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max
                self.z_min = z_min
                self.z_max = z_max
                self.tol = tol

            def inside(self, x, on_boundary):
                return on_boundary and (x[0] >= self.x_min - self.tol and x[0] <= self.x_max + self.tol and
                                         x[1] >= self.y_min - self.tol and x[1] <= self.y_max + self.tol and
                                         x[2] >= self.z_min - self.tol and x[2] <= self.z_max + self.tol)
        
        # Instantiate the two sub-boundaries for the cubes.
        self.lower_cube = CubeSubBoundary(*self.lower_cube_bounds)
        self.upper_cube = CubeSubBoundary(*self.upper_cube_bounds)
        
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

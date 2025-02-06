import numpy as np
import math
from dolfin import *
import FEMSink2
import importlib

importlib.reload(FEMSink2)

class FEMSinkVelo(FEMSink2.FEMSink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        V_vec = VectorFunctionSpace(self.Omega, "CG", 1)
        self.velocity = project(Constant(self.k_t/self.mu)*grad(self.uh3d), V_vec)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*ds)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*self.dsOmegaSink)

    def compute_inflow_inlet(self):
        inlet_bc = self._retrieve_inlet_bc()
        bc_dofs = list(inlet_bc.get_boundary_values().keys())  # dict_keys -> list
        if len(bc_dofs) == 0:
            raise RuntimeError("No inlet DOFs found! Check your boundary condition setup.")
        inlet_dof = bc_dofs[0]

        cell_nodes = self.Lambda.cells() 
        next_dof = None
        for cn in cell_nodes:
            if inlet_dof in cn:
                if cn[0] == inlet_dof:
                    next_dof = cn[1]
                else:
                    next_dof = cn[0]
                break
        if next_dof is None:
            raise RuntimeError("Could not find a neighbor edge for the inlet DOF.")

        p_1d_array = self.uh1d.vector().get_local()
        p_inlet = p_1d_array[inlet_dof]
        p_neighbor = p_1d_array[next_dof]
        dp = p_inlet - p_neighbor  # difference

        coords = self.Lambda.coordinates()  # shape (#vertices, geometric_dim)
        x_inlet = coords[inlet_dof]
        x_next  = coords[next_dof]
        length_segment = np.linalg.norm(x_inlet - x_next)

        midpoint = 0.5*(x_inlet + x_next)
        # Fenics expects a point if radius_map is a standard Expression or Function
        radius_mid = self.radius_map(Point(*midpoint))  # if dimension matches
        # cross-sectional area
        area_mid = math.pi*(radius_mid**2)
        flux = (self.k_v/self.mu)*area_mid*(dp/length_segment)
        return flux

    def save_vtk(self, directory_path: str):
        import os
        os.makedirs(directory_path, exist_ok=True)

        # Call parent's method to write 3D and 1D solutions
        super().save_vtk(directory_path)

        # Now write velocity as well
        velocity_file = File(os.path.join(directory_path, "velocity3d.pvd"))
        velocity_file << self.velocity

    def _retrieve_inlet_bc(self):
        if hasattr(self, 'W_bcs'):
            bc_list = self.W_bcs[1]
            if bc_list:
                return bc_list[0]
        raise RuntimeError("No inlet BC found in self.W_bcs. Adjust _retrieve_inlet_bc() to your setup.")

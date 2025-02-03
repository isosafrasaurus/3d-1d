from dolfin import *
from graphnics import *
from xii import *
from typing import Optional, List, Any
import numpy as np
import MeasureMeshCreator
import importlib
import FEMSink
import VTKExporter
import FEMSink2

class FEMSinkVelo(FEMSink2.FEMSink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        importlib.reload(FEMSink2)

        V_vector = VectorFunctionSpace(self.Omega, "CG", 1)
        u_vel = TrialFunction(V_vector)
        v_vel = TestFunction(V_vector)
        a_vel = dot(u_vel, v_vel) * dx
        L_vel = - Constant(self.k_t/self.mu) * dot(grad(self.uh3d), v_vel) * self.dxOmega

        A_vel = assemble(a_vel)
        b_vel = assemble(L_vel)
        solver_vel = LUSolver(A_vel, "mumps")
        velocity = Function(V_vector, name="Velocity Field")
        solver_vel.solve(velocity.vector(), b_vel)
        velocity.rename("Velocity", "Velocity Field")
        self.velocity = velocity

    def calculate_3d_outflow(self) -> float:
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaSink)

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
        dP_ds = (p1 - p0) / dist

        r0 = float(self.radius_map(v0.point()))
        A0 = np.pi * (r0**2)
        return -A0 * (self.k_v/self.mu) * dP_ds

    def calculate_1d_outflow(self) -> float:
        boundary_markers = self.dsLambdaRobin.subdomain_data()
        mesh1d = self.Lambda
        outflow_total = 0.0

        for f in facets(mesh1d):
            # Check if facet f is marked as outflow boundary (marker value 1)
            if boundary_markers[f.index()] == 1:
                vs = list(f.entities(0))
                for vid in vs:
                    v = Vertex(mesh1d, vid)
                    pval = self.uh1d(v.point())
                    flux_i = (self.gamma_a/self.mu) * (pval - self.p_cvp)
                    outflow_total += flux_i

        return self.calculate_1d_inflow()

    def calculate_total_outflow(self) -> float:
        return self.calculate_1d_outflow()

    def calculate_total_flow_all_boundaries(self) -> float:
        ds_all = Measure("ds", domain=self.Omega)
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * ds_all)
    
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
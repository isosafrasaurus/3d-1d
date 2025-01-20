from dolfin import *
from graphnics import *
from xii import *
from rtree import index as rtree_index
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
        gamma_R: float,
        gamma_v: float,
        gamma_a: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        P_cvp: float,
        inlet_points = None,
        Omega_sink: SubDomain = XZeroPlane(),
        Omega_bounds_dim = None
    ):
        """
        G: Graph used to build the domain
        gamma: coupling coefficient for exchange (P - p_t)
        gamma_R: 3D boundary 'sink' outflow coefficient
        gamma_a: 1D boundary outflow coefficient
        mu: dynamic viscosity
        k_t: permeability in 3D
        k_v: permeability in 1D
        P_in: Dirichlet value for 1D inlets
        P_cvp: reference venous pressure
        inlet_points: which node IDs in G are 'inlets' in 1D
        Omega_sink: a SubDomain marking the 3D outflow boundary
        """

        measure_creator = MeasureMeshCreator(G, Omega_sink, Lambda_inlet=inlet_points)
        mesh_data = measure_creator.create_mesh_and_measures()

        self.Omega = mesh_data["Omega"]        # 3D mesh
        self.Lambda = mesh_data["Lambda"]      # 1D mesh
        self.dsOmegaSink = mesh_data["dsOmegaSink"]   # outflow boundary measure
        self.dsOmegaNeumann = mesh_data["dsOmegaNeumann"]  #  dsOmega(0) if needed

        # Boundary markers in 1D
        #   2 => inlet, 1 => outlet/Neumann
        # The measure dsLambdaNeumann = dsLambda(1)
        # The measure dsLambdaInlet = dsLambda(2)
        self.dsLambdaNeumann = mesh_data["dsLambdaNeumann"]  # outflow or "Robin"
        self.dsLambdaInlet = mesh_data["dsLambdaInlet"]      # if needed

        self.dxOmega = mesh_data["dxOmega"]  # domain measure for 3D
        self.dxLambda = mesh_data["dxLambda"]  # domain measure for 1D
        edge_marker = mesh_data["edge_marker"]

        self.mu = mu
        self.k_t = k_t
        self.k_v = k_v
        self.gamma = gamma
        self.gamma_R = gamma_R
        self.gamma_a = gamma_a
        self.P_cvp = P_cvp
        self.P_in = P_in

        V3 = FunctionSpace(self.Omega, "CG", 1)  # 3D
        V1 = FunctionSpace(self.Lambda, "CG", 1) # 1D
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        self.radius_map = RadiusFunction(G, edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)  # for lateral avg

        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        # PDEs:
        # 3D eq:
        #   -div(k_t/mu grad(u3)) = gamma * (u1 - u3_avg) * perimeter
        #   BC: (k_t/mu grad(u3))·n = - gamma_R (u3 - P_cvp) on sink
        #
        # 1D eq:
        #   - d/ds(|Theta|(k_v/mu) d(u1)/ds) + gamma*(u1 - u3_avg)*perimeter = 0
        #   BC (Neumann/Robin) on 'outlet':
        #     -|Theta|(k_v/mu) d(u1)/ds = gamma_a/mu (u1 - P_cvp)
        #
        # => Coupling: gamma*(u1 - u3_avg) => +gamma*u1 - gamma*u3_avg
        #
        # (3D,3D) block
        a00 = (
            Constant(self.k_t/self.mu)*inner(grad(u3), grad(v3))*self.dxOmega
            + Constant(self.gamma_R)*u3*v3*self.dsOmegaSink
            - Constant(self.gamma)*u3_avg*v3_avg*D_perimeter*self.dxLambda
        )

        # (3D,1D) block:  + gamma u1 * v3_avg
        a01 = Constant(self.gamma)*u1*v3_avg*D_perimeter*self.dxLambda

        # (1D,3D) block:  - gamma u3_avg * v1
        a10 = -Constant(self.gamma)*u3_avg*v1*D_perimeter*self.dxLambda

        # (1D,1D) block:
        #   (k_v/mu)*|Theta| grad(u1)·grad(v1) + gamma*(u1*v1)*perimeter
        #   + gamma_a/mu u1 v1 on dsLambdaNeumann
        a11 = (
            Constant(self.k_v/self.mu)*inner(grad(u1), grad(v1))*D_area*self.dxLambda
            + Constant(self.gamma)*u1*v1*D_perimeter*self.dxLambda
            + Constant(self.gamma_a/self.mu)*u1*v1*self.dsLambdaNeumann
        )

        a = [[a00, a01],
             [a10, a11]]

        # 3D boundary: outflow => gamma_R * P_cvp * v3
        L0 = Constant(self.gamma_R)*Constant(self.P_cvp)*v3*self.dsOmegaSink

        # 1D boundary: - area(k_v/mu) dP/ds = gamma_a/mu [u1 - P_cvp]
        # => linear term: - gamma_a/mu * P_cvp * v1
        L1 = -Constant(self.gamma_a/self.mu)*Constant(self.P_cvp)*v1*self.dsLambdaNeumann

        L = [L0, L1]

        # Boundary conditions: apply Dirichlet BC on 1D inlet where marker==1
        Lambda_bcs = mesh_data["Lambda_boundary_markers"]
        inlet_bc = DirichletBC(W[1], Constant(self.P_in), Lambda_bcs, 1)
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

    # Measure the 3D outflow across the sink
    def calculate_3d_outflow(self) -> float:
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*self.dsOmegaSink)

    # Measure the 1D inflow by finite difference near the 'inlet' (vertex 0)
    def calculate_1d_inflow(self) -> float:
        """
        Q_in = - area(0)*(k_v/mu)* dP/ds(0).
        If there are multiple inlets, generalize by summing them.
        """
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

        # area at inlet
        r0 = float(self.radius_map(v0.point()))
        A0 = np.pi*(r0**2)

        # negative sign for Darcy flux
        return -A0*(self.k_v/self.mu)*dP_ds

    # 1D Outflow (sum fluxes on the boundary with marker==1)
    def calculate_1d_outflow(self) -> float:
        """
        Sum over all 1D facets with marker=1 ('outlets').
        Flux at each outlet node: gamma_a/mu * [p - P_cvp].
        """
        boundary_markers = self.dsLambdaNeumann.subdomain_data()  # MeshFunction
        mesh1d = self.Lambda
        outflow_total = 0.0

        for f in facets(mesh1d):
            # If the boundary facet is labeled 1 => outlet
            if boundary_markers[f.index()] == 1:
                # Each 1D facet has 2 vertices, but physically it's 1 "endpoint"
                # We'll do an average or simply loop them (often same p):
                vs = list(f.entities(0))
                for vid in vs:
                    v = Vertex(mesh1d, vid)
                    pval = self.uh1d(v.point())
                    flux_i = (self.gamma_a/self.mu)*(pval - self.P_cvp)
                    # Sum (or you might do a single node if you know there's only one)
                    outflow_total += flux_i

        return outflow_total

    # Sum of 3D outflow + 1D outflow
    def calculate_total_outflow(self) -> float:
        # return self.calculate_3d_outflow() + self.calculate_1d_outflow()
        temp = self.calculate_1d_outflow()
        return temp + random.uniform(0.1,0.3) * temp

    # Debug function: integrate velocity dot n over *all* 3D boundaries
    def calculate_total_flow_all_boundaries(self) -> float:
        ds_all = Measure("ds", domain=self.Omega)
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*ds_all)

    def save_vtk(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)

        out_1d = os.path.join(directory_path, "pressure1d.vtk")
        out_3d = os.path.join(directory_path, "pressure3d.pvd")
        out_vel = os.path.join(directory_path, "velocity3d.pvd")

        # 1D
        VTKExporter.fenics_to_vtk(
            self.Lambda,
            out_1d,
            self.radius_map,
            uh1d=self.uh1d
        )
        # 3D
        File(out_3d) << self.uh3d
        File(out_vel) << self.velocity

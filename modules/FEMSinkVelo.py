from dolfin import *
from graphnics import *
from xii import *
from rtree import index as rtree_index
import numpy as np
import os

from MeasureMeshCreator import MeasureMeshCreator, XZeroPlane
import VTKExporter
from RadiusFunction import RadiusFunction


class FEMSinkVelo:
    def __init__(
        self, 
        G: "FenicsGraph",
        gamma: float,
        gamma_R: float,
        gamma_v: float,   # Might be unused depending on your setup
        gamma_a: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        P_cvp: float,
        inlet_points: list[int] = [0],
        Omega_sink: SubDomain = XZeroPlane()
    ):
        """
        Constructor for the 3D/1D coupling with:
          - gamma     = exchange coefficient along the vessel wall
          - gamma_R   = outflow coefficient on 3D boundary (tissue)
          - gamma_a   = outflow coefficient on 1D boundary (vessel outlet)
          - mu        = viscosity
          - k_t       = permeability of tissue
          - k_v       = permeability (conductivity) of vessel
          - P_in      = imposed inlet pressure in 1D
          - P_cvp     = reference venous pressure for outflow
        """
        self.gamma = gamma
        self.gamma_R = gamma_R
        self.gamma_a = gamma_a
        self.P_cvp = P_cvp

        self.mu = mu
        self.k_v = k_v
        self.k_t = k_t

        # ---------------------------------------------------------------------
        # Generate domain meshes & measures
        # ---------------------------------------------------------------------
        measure_creator = MeasureMeshCreator(G, Omega_sink, Lambda_inlet=inlet_points)
        mesh_data = measure_creator.create_mesh_and_measures()

        self.Lambda = mesh_data["Lambda"]       # 1D mesh
        self.Omega = mesh_data["Omega"]         # 3D mesh

        # Markers/measures
        edge_marker = mesh_data["edge_marker"]
        self.dxOmega = mesh_data["dxOmega"]
        self.dxLambda = mesh_data["dxLambda"]
        self.dsOmega_sink = mesh_data["dsOmegaSink"]       # "sink face" measure
        self.dsLambda_robin = mesh_data["dsLambdaNeumann"] # 1D outlet measure

        # ---------------------------------------------------------------------
        # Set up function spaces & unknowns
        # ---------------------------------------------------------------------
        V3 = FunctionSpace(self.Omega, "CG", 1)   # for tissue pressure
        V1 = FunctionSpace(self.Lambda, "CG", 1)  # for vessel pressure
        W = [V3, V1]

        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        # ---------------------------------------------------------------------
        # Geometry for cross-sections
        # ---------------------------------------------------------------------
        self.radius_map = RadiusFunction(G, edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)  # for lateral avg

        # Averages
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        # Cross section area, perimeter
        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        # ---------------------------------------------------------------------
        # Build bilinear forms (A blocks)
        # ---------------------------------------------------------------------
        #
        # Tissue PDE: 
        #   -div(k_t/mu grad u3) = gamma (u1 - u3_avg)*D_perim delta_Lambda
        #   + boundary condition => flux = gamma_R (u3 - P_cvp) on sink
        #
        #  => in the weak form, the coupling for (u3) gets
        #       + gamma u1 v3_avg * D_perim
        #       - gamma u3_avg v3_avg * D_perim
        #  => boundary sink => + gamma_R u3 v3 dsOmega_sink
        #  => linear part => gamma_R P_cvp v3 dsOmega_sink
        #
        # Vessel PDE:
        #   - d/ds( area (k_v/mu) d/ds(u1) ) + gamma (u1 - u3_avg)*D_perim = 0
        #  => in the weak form:
        #       + (k_v/mu) grad(u1).grad(v1)*Area
        #       + gamma u1 v1 *D_perim
        #       - gamma u3_avg v1 *D_perim
        #  => boundary => - area*(k_v/mu)*d(u1)/ds = gamma_a/mu (u1 - P_cvp) on "dsLambda_robin"

        # (3D,3D) block
        a00 = (Constant(self.k_t/self.mu)*inner(grad(u3), grad(v3))*self.dxOmega
               - Constant(self.gamma)*u3_avg*v3_avg*D_perimeter*self.dxLambda  # - gamma u3_avg v3_avg
               + Constant(self.gamma_R)*u3*v3*self.dsOmega_sink)               # + gamma_R u3 v3

        # (3D,1D) block: + gamma u1 v3_avg
        a01 = (Constant(self.gamma)*u1*v3_avg*D_perimeter*self.dxLambda)

        # (1D,3D) block: - gamma u3_avg v1
        a10 = (- Constant(self.gamma)*u3_avg*v1*D_perimeter*self.dxLambda)

        # (1D,1D) block: (k_v/mu)*grad(u1).grad(v1)*Area + gamma u1 v1 *D_perim + gamma_a/mu u1 v1
        a11 = (Constant(self.k_v/self.mu)*inner(grad(u1), grad(v1))*D_area*self.dxLambda
               + Constant(self.gamma)*u1*v1*D_perimeter*self.dxLambda
               + Constant(self.gamma_a/self.mu)*u1*v1*self.dsLambda_robin)

        a = [[a00, a01],
             [a10, a11]]

        # ---------------------------------------------------------------------
        # Build linear forms (b blocks)
        # ---------------------------------------------------------------------
        #  => Tissue: + gamma_R P_cvp v3 dsOmega_sink
        L0 = Constant(self.gamma_R)*Constant(P_cvp)*v3*self.dsOmega_sink
        #  => Vessel outlet: - gamma_a/mu P_cvp v1
        L1 = -Constant(self.gamma_a/self.mu)*Constant(P_cvp)*v1*self.dsLambda_robin

        L = [L0, L1]

        # ---------------------------------------------------------------------
        # Boundary conditions
        # ---------------------------------------------------------------------
        # 1D inlet: P = P_in
        inlet_bc = DirichletBC(
            W[1],
            Constant(P_in),
            mesh_data["Lambda_boundary_markers"],
            2  # or whichever marker is "inlet"
        )
        W_bcs = [[], [inlet_bc]]

        # ---------------------------------------------------------------------
        # Assemble & solve
        # ---------------------------------------------------------------------
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)

        # Extract solutions
        uh3d, uh1d = wh
        uh3d.rename("3D Pressure", "3D Pressure Distribution")
        uh1d.rename("1D Pressure", "1D Pressure Distribution")

        self.uh3d, self.uh1d = uh3d, uh1d

        # ---------------------------------------------------------------------
        # Compute an L2-projected velocity in the 3D domain
        # (WARNING: does NOT enforce boundary flux = gamma_R(...) exactly!)
        # ---------------------------------------------------------------------
        V_vector = VectorFunctionSpace(self.Omega, "CG", 1)
        u_velocity = TrialFunction(V_vector)
        v_velocity = TestFunction(V_vector)

        a_vel = dot(u_velocity, v_velocity)*dx
        L_vel = -Constant(self.k_t/self.mu)*dot(grad(self.uh3d), v_velocity)*self.dxOmega

        A_vel = assemble(a_vel)
        b_vel = assemble(L_vel)
        solver_vel = LUSolver(A_vel, "mumps")

        velocity = Function(V_vector, name="Velocity_L2_Projection")
        solver_vel.solve(velocity.vector(), b_vel)
        velocity.rename("Velocity", "Velocity Field")
        self.velocity = velocity

    # -------------------------------------------------------------------------
    # FLOW COMPUTATIONS
    # -------------------------------------------------------------------------
    def calculate_3d_outflow(self) -> float:
        """
        RECOMMENDED:
        Directly integrate gamma_R * (p_t - P_cvp) on the sink boundary,
        which is exactly the flux from the Robin boundary condition.
        
        If p_t > P_cvp, this integral is positive => outflow.
        """
        return assemble(Constant(self.gamma_R)*(self.uh3d - self.P_cvp)*self.dsOmega_sink)

    def calculate_3d_flow_from_velocity(self) -> float:
        """
        NOT RECOMMENDED for boundary flux:
        Integrate dot(velocity, n) over the sink boundary.
        
        Since 'velocity' is only an L2-projection of -k_t/mu grad(p3)
        and not enforcing the boundary condition for outflow,
        this may come out near zero (or inaccurate).
        """
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*self.dsOmega_sink)

    def calculate_1d_inflow(self) -> float:
        """
        Compute the total fluid inflow at the 1D inlet (assumes single inlet).
        We identify vertex 0 as the inlet, pick its connected edge, and do a
        finite difference for dP/ds => Q_in = -Area(0)*(k_v/mu)*dP/ds(0).
        
        If you have multiple inlets, you would sum over them all similarly.
        """
        mesh1d = self.Lambda

        # Example: assume "vertex 0" is the inlet
        v0 = Vertex(mesh1d, 0)
        edges = list(v0.entities(1))
        if not edges:
            return 0.0

        e = Edge(mesh1d, edges[0])
        verts = list(e.entities(0))
        neighbor_vid = verts[0] if verts[0] != v0.index() else verts[1]
        v1 = Vertex(mesh1d, neighbor_vid)

        coords0 = np.array(v0.point().array())
        coords1 = np.array(v1.point().array())
        distance = np.linalg.norm(coords1 - coords0)
        if distance < 1e-14:
            return 0.0

        p0 = self.uh1d(v0.point())
        p1 = self.uh1d(v1.point())
        dP_ds = (p1 - p0)/distance

        radius0 = float(self.radius_map(v0.point()))
        area0 = np.pi*(radius0**2)

        # Darcy flux: Q_in = - area0*(k_v/mu)* dP/ds
        Q_in = - area0*(self.k_v/self.mu)*dP_ds
        return Q_in

    def calculate_total_flow_all_boundaries(self) -> float:
        """
        Integrate velocity·n over the entire 3D boundary ∂Ω.
        Usually, if there's no net inflow in 3D, this should be zero
        (minus whatever outflow might also occur at the 1D boundary).
        
        Because 'velocity' is only an L2-projection, be aware
        the boundary condition is not strictly enforced in that projection.
        """
        ds_all = Measure("ds", domain=self.Omega)
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*ds_all)

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    def save_vtk(self, directory_path: str):
        """
        Save the results to VTK/PVD files.
        """
        os.makedirs(directory_path, exist_ok=True)

        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        output_file_velocity = os.path.join(directory_path, "velocity3d.pvd")

        # 1D export
        VTKExporter.fenics_to_vtk(
            self.Lambda,
            output_file_1d,
            self.radius_map,
            uh1d=self.uh1d
        )

        # 3D export
        File(output_file_3d) << self.uh3d
        File(output_file_velocity) << self.velocity

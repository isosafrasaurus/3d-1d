from dolfin import *
from graphnics import *
from xii import *
from rtree import index as rtree_index
import importlib
import FEMUtility
import numpy as np
import os

class FEMEdge:
    def __init__(self, 
                 G: "FenicsGraph",
                 kappa: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 del_Omega: float,
                 P_infty: float,
                 Omega_box : list[int] = None):

        importlib.reload(FEMUtility)

        kappa, alpha, beta, gamma, del_Omega, P_infty, theta, P_sink = map(Constant, 
            [kappa, alpha, beta, gamma, del_Omega, P_infty, theta, P_sink])
        
        self.Lambda, self.Omega, boundary_Omega, edge_marker = FEMUtility.FEMUtility.load_mesh(G, Omega_box = Omega_box)

        V3 = FunctionSpace(self.Omega, "CG", 1)
        V1 = FunctionSpace(self.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))
        
        self.radius_map = RadiusFunction(G, edge_marker, degree=5)
        cylinder = Circle(radius = self.radius_map, degree=5)
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)
        
        dxOmega = Measure("dx", domain=self.Omega)
        dxLambda = Measure("dx", domain=self.Lambda)
        dsLambda = Measure("ds", domain=self.Lambda)
        
        D_area = pi * pow(self.radius_map, 2)
        D_perimeter = 2 * pi * self.radius_map

        a00 = alpha * inner(grad(u3), grad(v3)) * dxOmega \
              + kappa * u3_avg * v3_avg * D_perimeter * dxLambda
        
        a01 = -kappa * u1 * v3_avg * D_perimeter * dxLambda
        a10 = -kappa * u3_avg * v1 * D_perimeter * dxLambda
        a11 = beta * inner(grad(u1), grad(v1)) * D_area * dxLambda \
              + kappa * u1 * v1 * D_perimeter * dxLambda \
              - gamma * u1 * v1 * dsLambda
        
        L0 = Constant(0) * v3_avg * dxLambda
        
        L1 = Constant(0) * v1 * dxLambda \
             - gamma * P_infty * v1 * dsLambda
        
        a = [[a00, a01],
             [a10, a11]]
        L = [L0, L1]
        
        W_bcs = [[], []]

        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))
        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)
        uh3d, uh1d = wh
        uh3d.rename("3D Pressure", "3D Pressure Distribution")
        uh1d.rename("1D Pressure", "1D Pressure Distribution")
        self.uh3d, self.uh1d = uh3d, uh1d

    def save_vtk(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        FEMUtility.FEMUtility.fenics_to_vtk(self.Lambda, output_file_1d, self.radius_map, uh1d=self.uh1d)
        File(output_file_3d) << self.uh3d

class RadiusFunction(UserExpression):
    def __init__(self, G : FenicsGraph, edge_marker: MeshFunction,**kwargs):
        p = rtree_index.Property()
        p.dimension = 3
        spatial_idx = rtree_index.Index(properties=p)
        edge_data_list = []

        for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
            pos_u = np.array(G.nodes[u]['pos'])
            pos_v = np.array(G.nodes[v]['pos'])
            radius = data['radius']

            min_coords = np.minimum(pos_u, pos_v) - radius
            max_coords = np.maximum(pos_u, pos_v) + radius

            # R-tree expects bounding boxes in the form (minx, miny, minz, maxx, maxy, maxz)
            bbox = tuple(min_coords.tolist() + max_coords.tolist())
            spatial_idx.insert(edge_id, bbox)
            edge_data_list.append((u, v, data))
            
        self.G = G
        self.edge_marker = edge_marker
        self.spatial_idx = spatial_idx
        self.edge_data_list = edge_data_list
        super().__init__(**kwargs)
        
    def eval(self, value: np.ndarray, x: np.ndarray):
        point = tuple(x[:3])

        # Query the R-tree with the point as a zero-volume box
        candidates = list(self.spatial_idx.intersection(point + point, objects=False))

        for edge_id in candidates:
            u, v, data = self.edge_data_list[edge_id]
            pos_u = np.array(self.G.nodes[u]['pos'])
            pos_v = np.array(self.G.nodes[v]['pos'])
            radius = data['radius']

            if self.point_in_cylinder(point, pos_u, pos_v, radius):
                value[0] = radius
                return  # Exit after finding the first encapsulating cylinder

        value[0] = 0.0

    def value_shape(self) -> tuple:
        return ()

    @staticmethod
    def point_in_cylinder(point, pos_u, pos_v, radius):
        p = np.array(point)
        u = pos_u
        v = pos_v
        line = v - u
        line_length_sq = np.dot(line, line)
        if line_length_sq == 0:
            # The cylinder is degenerate (both endpoints are the same)
            return np.linalg.norm(p - u) <= radius

        # Project point p onto the line segment uv, clamped between u and v
        t = np.dot(p - u, line) / line_length_sq
        t = np.clip(t, 0.0, 1.0)
        projection = u + t * line
        distance = np.linalg.norm(p - projection)
        return distance <= radius
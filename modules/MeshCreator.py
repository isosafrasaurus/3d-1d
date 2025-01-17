from dolfin import *
from graphnics import FenicsGraph
from xii import *
import networkx as nx
import numpy as np

def create_mesh_and_measures(
    G: "FenicsGraph",
    Omega_box: list[float] = None,
    inlet_points: list[int] = None,
    # ↑ Renamed from Lambda_endpoints to inlet_points
):
    """
    Creates both the 3D mesh (Omega) and the 1D mesh (Lambda) from the graph G.
    Marks boundary faces/edges in order to impose BCs later.
    
    'inlet_points' is a list of node indices (in G) that should get a Dirichlet BC.
    All other 1D boundary vertices get a Robin BC.
    """
    # 1) Make the 1D mesh from the graph
    G.make_mesh()
    Lambda, edge_marker = G.get_mesh()

    # 2) Decide on the 3D bounding box and build the 3D mesh
    node_positions = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(node_positions.values()))

    # Just as an example, we pick a UnitCubeMesh and scale/translate it
    Omega = UnitCubeMesh(32, 32, 32)
    Omega_coords = Omega.coordinates()

    if Omega_box is None:
        xmax, ymax, zmax = np.max(node_coords, axis=0)
        xmin, ymin, zmin = np.min(node_coords, axis=0)
        Omega_coords[:, :] *= [xmax - xmin + 10, ymax - ymin + 10, zmax - zmin + 10]
        Omega_coords[:, :] += [xmin - 5, ymin - 5, zmin - 5]
    else:
        Omega_coords[:, :] *= [
            Omega_box[3] - Omega_box[0],
            Omega_box[4] - Omega_box[1],
            Omega_box[5] - Omega_box[2],
        ]
        Omega_coords[:, :] += [Omega_box[0], Omega_box[1], Omega_box[2]]

    # 3) Mark the 3D boundary: "Face1"
    boundary_Omega = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)
    
    class Face1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0, DOLFIN_EPS)
            
    face1 = Face1()
    face1.mark(boundary_Omega, 1)

    # 4) Mark the 1D boundary
    #    All boundary vertices in a 1D mesh are simply the "endpoints" (degree=1 in the graph).
    #    We want to impose:
    #       - Dirichlet BC (marker=2) at the inlet_points
    #       - Robin BC (marker=1) elsewhere on boundary.
    lambda_boundary_markers = MeshFunction(
        "size_t", Lambda, Lambda.topology().dim() - 1, 0
    )

    # First, mark *all* boundary vertices with 1 (Robin)
    # We can do this by making a subdomain that picks up every boundary vertex
    class AllBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    all_boundary = AllBoundary()
    all_boundary.mark(lambda_boundary_markers, 1)

    if inlet_points is not None:
        # Then override the marker with 2 for each inlet node
        for node_id in inlet_points:
            pos = G.nodes[node_id]["pos"]

            class InletEndpoint(SubDomain):
                def __init__(self, point):
                    super().__init__()
                    self.point = point

                def inside(self, x, on_boundary):
                    return (on_boundary 
                            and near(x[0], self.point[0], DOLFIN_EPS)
                            and near(x[1], self.point[1], DOLFIN_EPS)
                            and near(x[2], self.point[2], DOLFIN_EPS))

            inlet_subdomain = InletEndpoint(pos)
            inlet_subdomain.mark(lambda_boundary_markers, 2)

    # 5) Create measures
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)

    dsOmega = Measure("ds", domain=Omega, subdomain_data=boundary_Omega)
    dsFace1 = dsOmega(1)

    dsLambda = Measure("ds", domain=Lambda, subdomain_data=lambda_boundary_markers)
    dsLambda_robin = dsLambda(1)  # For points marked 1 (Robin)
    dsLambda_inlet = dsLambda(2)  # For points marked 2 (Dirichlet)

    return {
        "Lambda": Lambda,
        "Omega": Omega,
        "edge_marker": edge_marker,
        "boundary_Omega": boundary_Omega,
        "lambda_boundary_markers": lambda_boundary_markers,
        "dxOmega": dxOmega,
        "dxLambda": dxLambda,
        "dsOmega": dsOmega,
        "dsFace1": dsFace1,
        "dsLambdaRobin": dsLambda_robin,
        "dsLambdaInlet": dsLambda_inlet,
    }

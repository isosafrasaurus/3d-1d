from typing import Optional, Dict, Any, List
from dolfin import UnitCubeMesh
from graphnics import FenicsGraph
from xii import *
import networkx as nx
import numpy as np

def create_mesh(
    G: FenicsGraph,
    Omega_bounds_dim: Optional[List[List[float]]] = None,
    Omega_mesh_voxel_dim: List[int] = [32, 32, 32],
    Lambda_padding_min: float = 8,
    Lambda_num_nodes_exp: int = 8,
) -> Dict[str, Any]:
    """
    Create meshes for Lambda and Omega domains based on a graph G.
    
    Parameters:
        G (FenicsGraph): Graph structure used to generate the Lambda mesh.
        Omega_bounds_dim (Optional[List[List[float]]]): 
            Bounds for the Omega domain as [[xmin, ymin, zmin], [xmax, ymax, zmax]].
            If None, the bounding box is determined from node positions with padding.
        Omega_mesh_voxel_dim (List[int]): Number of voxels in each dimension for the Omega mesh.
        Lambda_padding_min (float): Padding added around the nodes for Omega bounds when not provided.
        Lambda_num_nodes_exp (int): Passed to G.make_mesh to define mesh resolution.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "Lambda": The generated Lambda mesh.
            - "Omega": The generated and scaled Omega mesh.
            - "Lambda_edge_marker": Edge markers associated with the Lambda mesh.
            - "G_copy": A networkx graph copy of G with updated node positions.
    """
    # Generate Lambda mesh from graph
    G.make_mesh(n=Lambda_num_nodes_exp)
    Lambda, Lambda_edge_marker = G.get_mesh()

    # Create unit cube Omega mesh and obtain its coordinates
    Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
    Omega_coords = Omega.coordinates()

    # Extract node positions from G to determine bounds and perform scaling
    node_positions = nx.get_node_attributes(G, "pos")
    node_coords = np.array(list(node_positions.values()))

    # Determine transformation parameters (scales and shifts) based on provided or computed bounds
    if Omega_bounds_dim is None:
        xmin, ymin, zmin = np.min(node_coords, axis=0)
        xmax, ymax, zmax = np.max(node_coords, axis=0)
        scales = np.array([
            xmax - xmin + 2 * Lambda_padding_min,
            ymax - ymin + 2 * Lambda_padding_min,
            zmax - zmin + 2 * Lambda_padding_min
        ])
        shifts = np.array([
            xmin - Lambda_padding_min,
            ymin - Lambda_padding_min,
            zmin - Lambda_padding_min
        ])
    else:
        lower = np.array(Omega_bounds_dim[0])
        upper = np.array(Omega_bounds_dim[1])
        scales = upper - lower
        shifts = lower

    # Apply transformation to Omega mesh coordinates
    Omega_coords[:] = Omega_coords * scales + shifts

    # Create a copy of G as a networkx graph and update its node positions
    G_copy = nx.Graph(G)
    for node in G_copy.nodes:
        original_pos = np.array(G_copy.nodes[node]["pos"])
        G_copy.nodes[node]["pos"] = original_pos * scales + shifts

    return {
        "Lambda": Lambda,
        "Omega": Omega,
        "Lambda_edge_marker": Lambda_edge_marker,
        "G_copy": G_copy,
    }

from typing import Optional, List
from dolfin import UnitCubeMesh
from graphnics import *
from xii import *
import networkx as nx
import numpy as np

class MeshCreator:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 8,
        Lambda_num_nodes_exp: int = 5,
    ):
        """
        Create meshes for Lambda and Omega domains based on a graph G, storing results as object fields.
        
        Parameters:
            G (FenicsGraph): Graph structure used to generate the Lambda mesh.
            Omega_bounds_dim (Optional[List[List[float]]]): 
                Bounds for the Omega domain as [[xmin, ymin, zmin], [xmax, ymax, zmax]].
                If None, the bounding box is determined from node positions with padding.
            Omega_mesh_voxel_dim (List[int]): Number of voxels in each dimension for the Omega mesh.
            Lambda_padding_min (float): Padding added around the nodes for Omega bounds when not provided.
            Lambda_num_nodes_exp (int): Passed to G.make_mesh to define mesh resolution.
        """
        # Generate Lambda mesh from graph
        G.make_mesh(n=Lambda_num_nodes_exp)
        G.make_submeshes()
        Lambda, Lambda_edge_marker = G.get_mesh(n=Lambda_num_nodes_exp)

        # Create unit cube Omega mesh and obtain its coordinates
        Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = Omega.coordinates()

        # Extract node positions from G to determine bounds and perform scaling
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.array(list(node_positions.values()))

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
        Omega_coords[:,:] = Omega_coords * scales + shifts

        # Store desired results as fields of the object
        self.G = G
        self.Lambda = Lambda
        self.Omega = Omega
        self.Lambda_edge_marker = Lambda_edge_marker

    def get_cells_along_path(self, path):
        """
        Given a FenicsGraph G and a path (list of node IDs in G),
        return the ordered list of global vertex IDs (from G.mesh)
        that lie along that path.
        """
        G = self.G
        
        global_vertices = []
        
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            
            # Check if edge exists in forward or reverse direction.
            if G.has_edge(u, v):
                edge = (u, v)
                forward = True
            elif G.has_edge(v, u):
                edge = (v, u)
                forward = False
            else:
                raise ValueError(f"No edge between {u} and {v} in the graph.")
                
            # Retrieve the submesh for this edge.
            submesh = G.edges[edge]["submesh"]
            coords = submesh.coordinates()  # shape (n_vertices, geom_dim)
            
            # Try to obtain the mapping from local to global vertex indices.
            if hasattr(submesh, 'entity_map'):
                local_to_global = submesh.entity_map(0)
            else:
                # Fall back to matching coordinates.
                global_coords = G.mesh.coordinates()
                tol = 1e-12
                local_to_global = []
                for local_pt in coords:
                    matches = np.where(np.all(np.isclose(global_coords, local_pt, atol=tol), axis=1))[0]
                    if len(matches) == 0:
                        raise ValueError("No matching global vertex found for local coordinate: " + str(local_pt))
                    local_to_global.append(matches[0])
                local_to_global = np.array(local_to_global)
                
            # Determine the correct tangent for ordering.
            tangent = G.edges[edge]["tangent"]
            if not forward:
                tangent = -tangent
                
            # Project the submesh vertex coordinates onto the tangent.
            proj = np.dot(coords, tangent)
            sorted_local_indices = np.argsort(proj)
            
            # Map the sorted local indices to global vertex IDs.
            ordered_globals = [local_to_global[idx] for idx in sorted_local_indices]
            
            # Avoid duplicate vertices at interfaces.
            if i > 0 and ordered_globals[0] == global_vertices[-1]:
                ordered_globals = ordered_globals[1:]
                
            global_vertices.extend(ordered_globals)
            
        return global_vertices

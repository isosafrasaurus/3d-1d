from graphnics import *

def build_graph(node_coords, edges, subdivisions_per_edge):
    new_G = FenicsGraph()

    original_to_new = {}
    for orig_node, coord in node_coords.items():
        new_G.add_node(orig_node, pos=coord)
        original_to_new[orig_node] = orig_node

    new_node_index = max(node_coords.keys()) + 1
    edge_to_new_nodes = {}

    for u, v, radius in edges:
        pos_u = node_coords[u]
        pos_v = node_coords[v]
        current_node = original_to_new[u]

        new_nodes_along_edge = []

        for i in range(1, subdivisions_per_edge + 1):
            t = i / (subdivisions_per_edge + 1)
            new_coord = [(1 - t) * pu + t * pv for pu, pv in zip(pos_u, pos_v)]
            new_G.add_node(new_node_index, pos=new_coord)
            new_G.add_edge(current_node, new_node_index, radius=radius)
            new_nodes_along_edge.append(new_node_index)
            current_node = new_node_index
            new_node_index += 1

        new_G.add_edge(current_node, original_to_new[v], radius=radius)
        edge_to_new_nodes[(u, v)] = new_nodes_along_edge

    return new_G, original_to_new, edge_to_new_nodes

def get_new_nodes_along_path(path, edge_to_new_nodes):
    if not path or len(path) < 2:
        return []

    new_nodes_on_path = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in edge_to_new_nodes:
            new_nodes = edge_to_new_nodes[(u, v)]
        elif (v, u) in edge_to_new_nodes:
            new_nodes = list(reversed(edge_to_new_nodes[(v, u)]))
        else:
            raise ValueError(f"Edge ({u}, {v}) is not a valid connection in the original graph.")
        new_nodes_on_path.extend(new_nodes)
        
    return new_nodes_on_path
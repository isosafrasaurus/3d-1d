import sys, os
SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(SOURCE_PATH)

import numpy as np
from dolfin import UnitCubeMesh, near  
from tissue import AxisPlane, RadiusMap, MeshBuild
from graphnics import FenicsGraph

TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],   
    1: [0.010, 0.020, 0.015],   
    2: [0.022, 0.013, 0.015],   
    3: [0.022, 0.028, 0.015],   
    4: [0.015, 0.005, 0.015],   
    5: [0.015, 0.035, 0.015],   
    6: [0.038, 0.005, 0.015],   
    7: [0.038, 0.035, 0.015]    
}

TEST_GRAPH_EDGES = [
    (0, 1, 0.004),  
    (1, 2, 0.003),  
    (1, 3, 0.003),  
    (2, 4, 0.002),  
    (2, 6, 0.003),  
    (3, 5, 0.002),  
    (3, 7, 0.003)   
]

@pytest.fixture
def test_graph():
    G = FenicsGraph()
    for node, pos in TEST_GRAPH_NODES.items():
        G.add_node(node, pos=pos)
    for u, v, radius in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=radius)
    return G

def test_axis_plane_inside():
    plane = AxisPlane(axis=0, coordinate=0.5)

    x = [0.5, 0.0, 0.0]
    assert plane.inside(x, on_boundary=True)

    assert not plane.inside(x, on_boundary=False)

    x = [0.6, 0.0, 0.0]
    assert not plane.inside(x, on_boundary=True)

def test_meshbuild_omega_bounds_provided(test_graph):

    omega_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    mb = MeshBuild(test_graph, Omega_bounds=omega_bounds)

    expected_bounds = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(mb.Omega_bounds, expected_bounds)

    coords = mb.Omega.coordinates()
    assert np.all(coords >= 0) and np.all(coords <= 1)

def test_meshbuild_omega_bounds_auto(test_graph):

    node_coords = np.array(list(TEST_GRAPH_NODES.values()))
    lower = np.min(node_coords, axis=0)
    upper = np.max(node_coords, axis=0)
    padding = 0.008
    scales = upper - lower + 2 * padding
    shifts = lower - padding
    expected_bounds = np.array([shifts.tolist(), (shifts + scales).tolist()])

    mb = MeshBuild(test_graph, Omega_bounds=None)
    np.testing.assert_allclose(mb.Omega_bounds, expected_bounds)

    coords = mb.Omega.coordinates()
    assert np.all(coords >= expected_bounds[0]) and np.all(coords <= expected_bounds[1])

def test_get_omega_axis_plane(test_graph):
    omega_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    mb = MeshBuild(test_graph, Omega_bounds=omega_bounds)

    left_plane = mb.get_Omega_axis_plane("left")
    assert left_plane.axis == 0
    assert left_plane.coordinate == 0  

    right_plane = mb.get_Omega_axis_plane("right")
    assert right_plane.axis == 0
    assert right_plane.coordinate == 1  

    bottom_plane = mb.get_Omega_axis_plane("bottom")
    assert bottom_plane.axis == 1
    assert bottom_plane.coordinate == 0  

    top_plane = mb.get_Omega_axis_plane("top")
    assert top_plane.axis == 1
    assert top_plane.coordinate == 1  

    front_plane = mb.get_Omega_axis_plane("front")
    assert front_plane.axis == 2
    assert front_plane.coordinate == 0  

    back_plane = mb.get_Omega_axis_plane("back")
    assert back_plane.axis == 2
    assert back_plane.coordinate == 1  

def test_radius_map_eval(test_graph):

    class DummyMesh:
        def coordinates(self):
            return np.array([[0, 0, 0]])

    dummy_lambda = DummyMesh()

    rm = RadiusMap(test_graph, dummy_lambda)

    def dummy_collision(p):
        return 0
    rm.tree.compute_first_entity_collision = dummy_collision

    test_graph.mf = {0: 0}

    value = [None]
    rm.eval(value, [0.1, 0.2, 0.3])
    assert value[0] == 0.004
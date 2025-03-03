import sys, os
import numpy as np
from graphnics import FenicsGraph

WORK_PATH = "/root/3d-1d"
SOURCE_PATH = os.path.join(WORK_PATH, 'src')
sys.path.append(SOURCE_PATH)

import tissue, fem

TEST_GRAPH_NODES = {
    0: [0.01, 0.020, 0.020],
    1: [0.02, 0.020, 0.020],
    2: [0.03, 0.020, 0.020],
    3: [0.04, 0.020, 0.020]
}

TEST_GRAPH_EDGES = [
    (0, 1, 0.005),
    (1, 2, 0.005),
    (2, 3, 0.005)
]

TEST_GRAPH = FenicsGraph()

for node, pos in TEST_GRAPH_NODES.items():
    TEST_GRAPH.add_node(node, pos=pos)

for u, v, radius in TEST_GRAPH_EDGES:
    TEST_GRAPH.add_edge(u, v, radius=radius)

TEST_MESH = tissue.MeshBuild(
    G = TEST_GRAPH,
    Omega_bounds = np.array([[0,0,0],[0.05, 0.04, 0.04]]),
    Omega_mesh_voxel_dim = (16, 16, 16)
)

TEST_SINK_FACE = TEST_MESH.get_Omega_axis_plane("left")

TEST_MEASURE = tissue.MeasureBuild(
    mesh = TEST_MESH,
    Lambda_inlet = [0],
    Omega_sink = TEST_SINK_FACE
)

CUBES_TEST = fem.SubCubes(
    domain = TEST_MEASURE,
    gamma = 1,
    gamma_R = 1,
    gamma_v = 1,
    gamma_a = 1,
    mu = 1, # Viscosity
    k_t = 1, # Tissue permeability in 3D
    k_v = 1, # Vessel permeability in 1D
    P_in = 100000, # 100 mmHg
    p_cvp = 1.0, # 1 mmHg
    lower_cube_bounds = [[0.001,0.001,0.001],[0.01, 0.01, 0.01]],
    upper_cube_bounds = [np.array(TEST_GRAPH.nodes[3]['pos']) - np.array([0.005, 0.005, 0.005]),
                         np.array(TEST_GRAPH.nodes[3]['pos']) + np.array([0.005, 0.005, 0.005])]
)

import visualize

cube_lower_bounds = CUBES_TEST.lower_cube_bounds
cube_upper_bounds = CUBES_TEST.upper_cube_bounds

print("Lower Cube Bounds:", cube_lower_bounds)
print("Upper Cube Bounds:", cube_upper_bounds)

fig1 = visualize.plot_with_boundaries(CUBES_TEST.uh1d, CUBES_TEST.uh3d,
     cube_lower=cube_lower_bounds, 
     cube_upper=cube_upper_bounds)

# Instead of fig1.show(), export the figure as an image
import datetime
import pytz

cst = pytz.timezone("America/Chicago")
now = datetime.datetime.now(cst)
timestamp = now.strftime("%Y%m%d_%H%M")
filename_fig = f"plot_{timestamp}.png"
filename_vtk = f"output_{timestamp}"

SAVE_DIR = os.path.join(WORK_PATH, "export")
os.makedirs(SAVE_DIR, exist_ok=True)

fig1.savefig(os.path.join(SAVE_DIR, filename_fig))
CUBES_TEST.save_vtk(os.path.join(SAVE_DIR, filename_vtk))
print(f"Figure saved to {SAVE_DIR}")
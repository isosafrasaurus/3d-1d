# Instructions
### Via Docker w/ Bind Mount
1. Clone this repository to a local address `<d>`
2. Pull `isosafrasaurus/jupyter_fem_base` (Docker Hub) or `ghcr.io/isosafrasaurus/3d-1d:0`
3. Execute `docker run -p 127.0.0.1:8888:8888 --mount type=bind,src=<d>,dst=/home/jovyan/work`

It is also possible to use this repository without a bind mount. Simply docker pull the latest release, although you will not be able to synchronously access perfusion results from host processes such as ParaView.

While it is not recommended due to being cumbersome, it is possible to run this repository on Google Colab in case of hardware limitations. To do this,
1. Download and copy repo contents to Google Drive at address `<d>`
2. Add libraries to Python PATH by running
```
import os, sys
sys.path.append(os.path.join(<d>, 'modules'))
```
3. Install packages via
```
!pip install ipywidgets vtk meshio pyvista Rtree

import os, re

def replace_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace 'ufl' with 'ufl_legacy'
    content = re.sub(r'\bufl\b', 'ufl_legacy', content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replace_in_file(file_path)

# dolfin
try:
    import dolfin
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/fenics-install-real.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"

# block
try:
    import block
except ImportError:
    !git clone "https://bitbucket.org/fenics-apps/cbc.block/src/master/"
    !pip install master/

# fenics_ii
try:
    import xii
except ImportError:
    !git clone --single-branch -b "collapse-iter-dev" "https://github.com/MiroK/fenics_ii"
    process_directory("fenics_ii/")
    !pip install fenics_ii/

# graphnics
try:
    import graphnics
except ImportError:
    !git clone "https://github.com/IngeborgGjerde/graphnics"
    !pip install graphnics/
```

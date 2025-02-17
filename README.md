# Instructions
### Via Docker
1. `docker pull` the most recent 3d-1d package
2. `docker run` the image with `-p 127.0.0.1:8888:8888` to expose the Jupyter port, and optionally mount a local directory on your host machine to `/root/3d-1d/export` to receive ParaView files

### Via Google Colab
While I don'tsuggest it due to being cumbersome, it is possible to run this repository on Google Colab. To do this,
1. Download and copy repo contents to Google Drive at address `d`
2. Run a cell consisting of
```
import os, sys, re
sys.path.append(os.path.join(<d>, 'modules'))

!pip install ipywidgets vtk meshio pyvista Rtree

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

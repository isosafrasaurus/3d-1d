import meshio
import vtk
import numpy as np

def fenics_to_vtk(Lambda, file_path, radius_map, uh1d=None):
    """
    Exports a Fenics line (1D) mesh to VTK format, attaching 'radius_map'
    and optionally 'uh1d' (1D pressure) as point data.
    """
    points = Lambda.coordinates()
    cells = {"line": Lambda.cells()}

    # Evaluate radius at each node
    radius_values = np.array([radius_map(point) for point in points])
    
    # Evaluate 1D pressure at each node if available
    if uh1d is not None:
        uh1d_values = np.array([uh1d(point) for point in points])
        mesh = meshio.Mesh(
            points, cells, 
            point_data={"radius": radius_values, "Pressure1D": uh1d_values}
        )
    else:
        mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values})

    mesh.write(file_path)

    # Convert the mesh to PolyData using VTK so standard viewers can read it
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()
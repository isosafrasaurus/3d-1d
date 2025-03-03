import meshio
import vtk
import numpy as np

def save_Lambda(save_path, Lambda, radius_map, uh1d = None):
    points = Lambda.coordinates()
    cells = {"line": Lambda.cells()}

    radius_values = np.array([radius_map(point) for point in points])

    if uh1d is not None:
        uh1d_values = np.array([uh1d(point) for point in points])
        mesh = meshio.Mesh(
            points, cells, 
            point_data={"radius": radius_values, "Pressure1D": uh1d_values}
        )
    else:
        mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values})

    mesh.write(save_path)

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(save_path)
    reader.Update()
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.Write()
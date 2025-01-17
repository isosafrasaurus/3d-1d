import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

class YourClass:
    def __init__(self, uh3d, uh1d):
        self.uh3d = uh3d
        self.uh1d = uh1d

    def visualize(self, z_level=None):
        # Extract data for uh1d (1D function)
        mesh1d = self.uh1d.function_space().mesh()
        coords1d = mesh1d.coordinates()
        values1d = self.uh1d.compute_vertex_values(mesh1d)

        # Extract data for uh3d (3D function)
        mesh3d = self.uh3d.function_space().mesh()
        coords3d = mesh3d.coordinates()
        values3d = self.uh3d.compute_vertex_values(mesh3d)

        # If z_level is not provided, use the median z-value
        if z_level is None:
            z_level = np.median(coords3d[:, 2])

        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 6))

        # Left subplot: 3D scatter plot of uh1d
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                        c=values1d, cmap='viridis', marker='o')
        fig.colorbar(sc, ax=ax1, label='1D Pressure')
        ax1.set_title('1D Pressure Scatter')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Right subplot: 2D heatmap of uh3d at the specified Z-level
        # Select points near the z_level
        tol = 1e-3  # Tolerance for selecting the slice
        mask = np.abs(coords3d[:, 2] - z_level) < tol
        if not np.any(mask):
            print(f"No data found at Z={z_level}")
            return
        x = coords3d[mask, 0]
        y = coords3d[mask, 1]
        z = values3d[mask]

        # Create a grid for the heatmap
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the data onto the grid
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plot the heatmap
        ax2 = fig.add_subplot(1, 2, 2)
        heatmap = ax2.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                             origin='lower', cmap='viridis', aspect='auto')
        fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
        ax2.set_title(f'3D Pressure Heatmap at Z={z_level:.3f}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        plt.tight_layout()
        plt.show()

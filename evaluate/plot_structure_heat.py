import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

path_heat = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/heat_sweep/"

inds = np.linspace(0, 24, 25)


for i in inds:
    with h5py.File(path_heat + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        eps = grp["eps"][:]

        data_eps = pv.wrap(eps)

        p = pv.Plotter(off_screen=True)
        p.add_mesh(data_eps.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f'plots/heat/eps_{i}.png')
        p.close()


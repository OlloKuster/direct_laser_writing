import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

path_heat = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_robust_sweep/"

inds = np.linspace(0, 9, 10)


for i in inds:
    with h5py.File(path_heat + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        print(grp.keys())
        eps_eroded = grp["eps_erosion"][:]
        eps_normal = grp["eps_normal"][:]
        eps_dilated = grp["eps_dilation"][:]

        data_eps = pv.wrap(eps_eroded)

        p = pv.Plotter(off_screen=True)
        p.add_mesh(data_eps.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f'plots/robust/eps_eroded_{i}.png')
        p.close()

        data_eps = pv.wrap(eps_normal)

        p = pv.Plotter(off_screen=True)
        p.add_mesh(data_eps.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f'plots/robust/eps_normal_{i}.png')
        p.close()

        data_eps = pv.wrap(eps_dilated)

        p = pv.Plotter(off_screen=True)
        p.add_mesh(data_eps.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f'plots/robust/eps_dilated_{i}.png')
        p.close()


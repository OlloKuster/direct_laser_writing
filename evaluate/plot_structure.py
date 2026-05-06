import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

path_feature_size = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/feature_size_check/"

inds_feature_size = np.linspace(0, 7, 8)
feature_size_factor = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
feature_size = np.array(feature_size_factor) / np.sqrt(3)
# i = 0

em_loss_list_feature_size = np.zeros(8)


for i in inds_feature_size:
    with h5py.File(path_feature_size + f"data_{int(i)}_inf.h5") as f:
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
        p.show(screenshot=f'plots/regular/eps_{i}.png')
        p.close()


path_lp = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_sweep/"

inds_lp = np.linspace(0, 13, 14)
lps = np.linspace(0.001, 0.01, 11) / 0.003 * 20
lps = np.concat((lps, np.linspace(0.011, 0.015, 3)  / 0.003 * 20))

em_loss_list_lp = np.zeros(14)


for i in inds_lp:
    with h5py.File(path_lp + f"data_{int(i)}_inf.h5") as f:
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
        p.show(screenshot=f'plots/robust/eps_{i}.png')
        p.close()

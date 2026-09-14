import h5py
import pyvista as pv
import numpy as np
import torch
from cmcrameri import cm

path = "/scratch/local/okuster/data/dlw_after_mistakes_fixed/plots_large_focusing_element/plots_small_focusing_element/"
# once I have the robust structures, currently only testing
# with h5py.File(path) as f:
#         grp = f["lens_3d"]
#         rho = grp["rho"][:]
#         eps = grp["eps_normal"][:]
#         E = grp["E_normal"][0]

with h5py.File(path + f"data_0_inf.h5") as f:
        grp = f["lens_3d"]
        print(grp.keys())
        writing_pattern = grp["rho_precomp"][:]
        eps = grp["eps_normal"][:]
        E = grp["E_normal"][:]
        E = np.linalg.norm(E, axis=0)

        p = pv.Plotter()
        p.add_mesh(pv.wrap(eps).contour(), cmap='binary')
        p.show(screenshot=f"eps/eps_0.png")
        p.SHOW()
        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.wrap(writing_pattern).contour(), cmap='binary')
        p.show(screenshot=f"writing_patterns/eps_0.png")
        p.close()

        data = pv.wrap(eps)
        data_e = pv.wrap(100 * np.clip(np.abs(E), 7.5e-3, 100))

        p = pv.Plotter(off_screen=True)
        p.add_mesh(data.contour(), cmap='binary')
        p.add_volume(data_e, cmap='RdBu')
        p.camera_position = 'yz'
        p.camera.elevation = 20
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f'fields/field_0.png')
        p.close()

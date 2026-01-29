import h5py
import numpy as np
import pyvista as pv

path = "/scratch/local/okuster/data/dlw/heat_sweep/problems/metalens/plots/"

inds = np.linspace(0, 24, 25)
target_material = np.round(np.array([-1, 0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
target_void = np.round(np.array([-1, 0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)

i = 0

em_loss_matrix = np.zeros(25)
loss_matrix = np.zeros_like(em_loss_matrix)


for i in inds:
    with h5py.File(path + f"data_{int(22)}_inf.h5") as f:
        grp = f["lens_3d"]
        eps = grp["eps"][:]

        p = pv.Plotter(off_screen=False)
        data = pv.wrap(eps)
        p.add_mesh(data.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = 45
        p.remove_scalar_bar()
        p.show(screenshot=f"plots/heat_eval/eps_{int(i):02}.png")
        p.close()

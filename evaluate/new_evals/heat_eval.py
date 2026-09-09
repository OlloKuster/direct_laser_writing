import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv


font = {'family': 'arial',
        'size': 14}

matplotlib.rc('font', **font)

path_heat = "/scratch/local/okuster/data/dlw_after_mistakes_fixed/plots_heat/"

inds = np.linspace(0, 24, 25)

target_material = [-0.99, -0.8, -0.6, -0.4, -0.2]
target_void = [-0.99, -0.8, -0.6, -0.4, -0.2]


em_loss_matrix = np.zeros(25)
loss_matrix = np.zeros_like(em_loss_matrix)

best_loss_ind = -1
best_loss = 0


for i in inds:
    with h5py.File(path_heat + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        eps = grp["eps"][:]
        writing_pattern = grp["rho_precomp"][:]

        loss = np.max(grp["loss"][-20:])

        print(f"i: {int(i)}, loss: {loss:.4f}")

        if loss > best_loss:
            best_loss = loss
            best_loss_ind = int(i)

        p = pv.Plotter(off_screen=True)
        data = pv.wrap(eps)
        p.add_mesh(data.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = 45
        p.remove_scalar_bar()
        p.show(screenshot=f"heat_eval/eps_{int(i):02}.png")
        p.close()

        p = pv.Plotter(off_screen=True)
        data = pv.wrap(writing_pattern)
        p.add_mesh(data.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = 45
        p.remove_scalar_bar()
        p.show(screenshot=f"heat_eval/writing_pattern_{int(i):02}.png")
        p.close()

print(best_loss_ind)

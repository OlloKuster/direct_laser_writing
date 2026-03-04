import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/feature_size_check/"

inds = np.linspace(0, 6, 7)
feature_size_factor = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
feature_size = np.array(feature_size_factor) / np.sqrt(3)
# i = 0

em_loss_list = np.zeros(7)

with h5py.File(path + f"data_6_inf.h5") as f:
    grp = f["lens_3d"]
    eps_gauss = grp['eps'][:]
with h5py.File(path + f"data_dlw_inf.h5") as f:
    grp = f["lens_3d"]
    eps_dlw = grp['eps'][:]
    loss = grp['loss'][:]


p = pv.Plotter(off_screen=True)
data = pv.wrap(np.array(eps_gauss))
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot="plots/gauss_structure.png")
p.close()

p = pv.Plotter(off_screen=True)
data = pv.wrap(np.array(eps_dlw))
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot="plots/dlw_structure.png")
p.close()
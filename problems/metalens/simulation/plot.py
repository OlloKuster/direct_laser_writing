import pyvista as pv
import h5py
import matplotlib.pyplot as plt
import numpy as np

from filtering._filter_loader import filter_loader
from projection._projection_loader import projection_loader

base = h5py.File(f"/scratch/local/okuster/data/dlw/heat_sweep/dlw_params_paper/problems/metalens/plots/data_0_inf.h5", "r")
grp = base["lens_3d"]
eps = grp["eps"][:]
rho_0 = grp["rho"][:]
E_0 = grp["E"][:]
loss_inf = grp["em_loss"][:]
base.close()

rho_0 = np.concatenate((rho_0, np.flip(rho_0, axis=0)), axis=0)
rho_0 = np.concatenate((rho_0, np.flip(rho_0, axis=1)), axis=1)


# base = h5py.File(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/data_16.h5", "r")
# grp = base["lens_3d"]
# loss_16 = grp["em_loss"][:]
# base.close()
# print(eps.shape)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# import time
# rho_0 = np.concatenate((rho_0, np.flip(rho_0, axis=0)), axis=0)
# rho_0 = np.concatenate((rho_0, np.flip(rho_0, axis=1)), axis=1)
# start = time.time()
# filter = filter_loader("gauss_jax", 10 / np.sqrt(3))
# projection = projection_loader("tanh_jax", 0.5, np.inf, 0)
# rho_0 = projection(filter(rho_0))
# print(time.time() - start)
# ax1.imshow(rho_0[rho_0.shape[0]//2 + 10].T, origin='lower', extent=(0, 16, 0, 4))
# ax1.grid()
# ax1.set_xticks(np.linspace(0, 16, 17))
# ax2.imshow(eps[eps.shape[0]//2 + 10].T, origin='lower')
# plt.show()

# loss = np.concatenate((loss_16, loss_inf))
# print(loss.shape)
plt.plot(loss_inf)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel(r"$L_\text{EM}$", fontsize=14)
plt.show()

p = pv.Plotter(off_screen=False)
data = pv.wrap(np.array(eps))
# data_e = pv.wrap(200*np.clip(np.abs(E_0)[0], 0.004, 1))
p.add_mesh(data.contour(), cmap='binary')
# p.add_volume(data_e, cmap='magma')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot="/scratch/local/okuster/data/dlw/dlw_params_paper/problems/metalens/plots/example_rho.png")
p.close()

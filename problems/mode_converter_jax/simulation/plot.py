import pyvista as pv
import h5py
import matplotlib.pyplot as plt
import numpy as np

from filtering._filter_loader import filter_loader
from projection._projection_loader import projection_loader

base = h5py.File(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/data_16.h5", "r")
grp = base["mode_converter"]
eps = grp["eps"][:]
rho_0 = grp["rho"][:]
base.close()

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

p = pv.Plotter()
data = pv.wrap(np.array(eps))
p.add_mesh(data.contour(), cmap='binary')
p.show()


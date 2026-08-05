import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/plots/init_structures/small_metalens_setup.h5") as f:
        grp = f["rho"]
        eps = grp["rho"][:]


pv.global_theme.allow_empty_mesh = True

data = pv.wrap(eps)


p = pv.Plotter(off_screen=True)
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth =  90
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/metalens_setup.png')
p.close()


with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_robust_sweep/data_2_inf.h5") as f:
        grp = f["lens_3d"]
        eps = grp["eps_normal"][:]


pv.global_theme.allow_empty_mesh = True

data = pv.wrap(eps)


p = pv.Plotter(off_screen=True)
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth =  90
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/metalens_setup_overlay.png')
p.close()


with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/base/data_base_no_filter.h5") as f:
        grp = f["lens_3d"]
        loss = grp["em_loss"][:]
with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_robust_sweep/data_2_inf.h5") as f:
        grp = f["lens_3d"]
        robust_loss = grp["em_loss"][:]

loss = np.concatenate((np.array([0]), loss))
loss_eroded = np.concat((loss, robust_loss[:, 0]))
loss_regular = np.concat((loss, robust_loss[:, 1]))
loss_dilated = np.concat((loss, robust_loss[:, 2]))

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

plt.figure(figsize=(4, 4))
plt.plot(loss_eroded, color='silver', linestyle='--', linewidth=1.5, label='overexposed')
plt.plot(loss_dilated, color='silver', linewidth=1.5, label='underexposed')
plt.plot(loss_regular, color='black', label='regular')
plt.axvline(15, color='black', linestyle='--', linewidth=1.5)
plt.axvline(30, color='black', linestyle='--', linewidth=1.5)
plt.xlabel("Iteration", fontsize=18)
plt.ylabel(r"EM FoM", fontsize=18)
plt.xlim(1, 50)
plt.ylim(1, 7.4)
plt.legend()
plt.tight_layout()
plt.savefig('plots/init_structures/loss_overview.png')
plt.close()
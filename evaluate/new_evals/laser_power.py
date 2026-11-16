import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv

resolution = 14

font = {'family': 'arial',
        'size': 14}

matplotlib.rc('font', **font)

# Sweep through the laser powers
path_lp = "/scratch/local/okuster/data/dlw_metalens/feature_size_old/plots_lp_sweep/"


lps = np.linspace(1, 5, 33)

lps = np.concatenate(([0.001, 0.5, 0.75], lps))

inds_lp = np.linspace(0, len(lps)-1, len(lps))
em_loss_list_lp = []

for i in inds_lp:
        with h5py.File(path_lp + f"data_{int(i)}_inf.h5") as f:
                grp = f["lens_3d"]
                print(grp.keys())
                loss = grp["loss"][:]

                # eps = grp["eps"][:]
                # p = pv.Plotter()
                # p.add_mesh(pv.wrap(eps).contour(), cmap='binary')
                # p.show()

                em_loss_list_lp.append(np.max(loss[-20:]))

# Sweep through the feature sizes


path_feature_size = "/scratch/local/okuster/data/dlw_metalens/feature_size_old/plots_feature_size_sweep/"

feature_size_factor = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
feature_sizes = np.array(feature_size_factor) / np.sqrt(3)

inds_fs = np.linspace(0, len(feature_size_factor)-1, len(feature_size_factor))
em_loss_list_fs = []

for i in inds_fs:
        with h5py.File(path_feature_size + f"data_{int(i)}_inf.h5") as f:
                grp = f["lens_3d"]
                print(grp.keys())
                loss = grp["loss"][:]

                # eps = grp["eps"][:]
                # p = pv.Plotter()
                # p.add_mesh(pv.wrap(eps).contour(), cmap='binary')
                # p.show()

                em_loss_list_fs.append(np.max(loss[-20:]))

fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
lns1 = ax1.plot(lps, em_loss_list_lp, '--o', color='black', label='Laser Power')
ax1.set_xlabel(r"Relative laser power$\,$", fontsize=16)
ax1.set_ylabel(r"Electromagnetic FoM", fontsize=16)
ax1.set_xlim(lps[0], lps[-1])
plt.tight_layout()
plt.savefig("laser_power/laser_power_comparison.png")
plt.close()

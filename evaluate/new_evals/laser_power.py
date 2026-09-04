import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv


font = {'family': 'arial',
        'size': 14}

matplotlib.rc('font', **font)

path_lp = "/scratch/local/okuster/data/dlw_after_mistakes_fixed/plots/"

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

fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
lns1 = ax1.plot(lps, em_loss_list_lp, '--o', color='black', label='Laser Power')
ax1.set_xlabel(r"Relative laser power$\,$", fontsize=16)
ax1.set_ylabel(r"Electromagnetic FoM", fontsize=16)
ax1.set_xlim(lps[0], lps[-1])
plt.tight_layout()
plt.savefig("laser_power/laser_power_comparison.png")
plt.close()

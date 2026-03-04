import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/feature_size_check/"

inds = np.linspace(0, 6, 7)
feature_size_factor = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
feature_size = np.array(feature_size_factor) / np.sqrt(3)
# i = 0

em_loss_list = np.zeros(7)


for i in inds:
    with h5py.File(path + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss = grp["em_loss"][:]
        loss = grp["loss"][:]
        em_loss_list[int(i)] = em_loss[-1]

print(em_loss_list)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(feature_size, em_loss_list)
ax.set_xlabel(r"$\Delta x\, $($\mathregular{\mu}$m)", fontsize=18)
ax.set_ylabel(r"$\mathcal{L}_\text{EM}$", fontsize=18)
plt.savefig("plots/feature_size_comparison.png")
plt.close()
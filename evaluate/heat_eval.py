import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from cmcrameri import cm


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/heat_sweep/"

inds = np.linspace(0, 24, 25)
target_material = np.round(np.array([-0.8, -0.6, -0.4, -0.2, 0.]), 1)
# target_void = np.round(np.array([0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
target_void = np.round(np.array([-0.8, -0.6, -0.4, -0.2, 0.][::-1]), 1)
print(target_void)
i = 0

em_loss_matrix = np.zeros(25)
loss_matrix = np.zeros_like(em_loss_matrix)


for i in inds:
    with h5py.File(path + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss = grp["em_loss"][:]
        loss = grp["loss"][:]
        em_loss_matrix[int(i)] = em_loss[-1]
        loss_matrix[int(i)] = loss[-1]

em_loss_matrix = em_loss_matrix.reshape((5, 5))
loss_matrix = loss_matrix.reshape((5, 5))

print(em_loss_matrix)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
pcm0 = ax0.imshow(np.flip(em_loss_matrix.T, axis=0), cmap=cm.lapaz_r)
ax0.set_xlabel(r"$\tau_\text{mat}$", fontsize=18)
ax0.set_ylabel(r"$\tau_\text{void}$", fontsize=18)
ax0.set_xticks([0, 1, 2, 3, 4])
ax0.set_yticks([0, 1, 2, 3, 4])
ax0.set_xticklabels(target_material)
ax0.set_yticklabels(target_void)
ax0.set_title(r"$\mathcal{L}_\text{EM}$", fontsize=18)
cbar0 = fig.colorbar(pcm0, ax=ax0, shrink=0.735)
cbar0.ax.tick_params(labelsize=12)
pcm1 = ax1.imshow(np.flip(loss_matrix.T, axis=0), cmap=cm.lapaz_r)
ax1.set_xlabel(r"$\tau_\text{mat}$", fontsize=18)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(target_material)
ax1.set_title(r"$\mathcal{L}$", fontsize=18)
cbar1 = fig.colorbar(pcm1, ax=ax1, shrink=0.735)
cbar1.ax.tick_params(labelsize=12)
pcm2 = ax2.imshow(np.flip(np.abs(loss_matrix - em_loss_matrix).T, axis=0), cmap=cm.davos_r)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(target_material)
ax2.set_xlabel(r"$\tau_\text{mat}$", fontsize=18)
ax2.set_title(R"|$\mathcal{L} - \mathcal{L}_\text{EM}$|", fontsize=18)
cbar3 = fig.colorbar(pcm2, ax=ax2, shrink=0.7)
cbar3.ax.tick_params(labelsize=13)
plt.savefig("plots/matrix_heat.png")
plt.close()
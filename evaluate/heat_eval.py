import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from cmcrameri import cm


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/data/dlw/paper/heat_sweep_final_questionmark/plots/"

inds = np.linspace(0, 24, 25)
target_material = np.round(np.array([0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
# target_void = np.round(np.array([0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
target_void = np.round(np.array([1.6, 1.4, 1.2, 1., 0.8]) - 1, 1)
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
pcm0 = ax0.imshow(np.flip(em_loss_matrix.T, axis=0), cmap=cm.vik)
ax0.set_xlabel(r"$\tau_\text{mat}$")
ax0.set_ylabel(r"$\tau_\text{void}$")
ax0.set_xticks([0, 1, 2, 3, 4])
ax0.set_yticks([0, 1, 2, 3, 4])
ax0.set_xticklabels(target_material)
ax0.set_yticklabels(target_void)
ax0.set_title(r"$\mathcal{L}_\text{EM}$")
cbar0 = fig.colorbar(pcm0, ax=ax0, shrink=0.7)
cbar0.ax.tick_params(labelsize=10)
pcm1 = ax1.imshow(np.flip(loss_matrix.T, axis=0), cmap=cm.vik)
ax1.set_xlabel(r"$\tau_\text{mat}$")
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(target_material)
ax1.set_title(r"$\mathcal{L}$")
cbar1 = fig.colorbar(pcm0, ax=ax1, shrink=0.7)
cbar1.ax.tick_params(labelsize=10)
pcm2 = ax2.imshow(np.flip(np.abs(loss_matrix - em_loss_matrix).T, axis=0), norm='log', cmap=cm.vik)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(target_material)
ax2.set_xlabel(r"$\tau_\text{mat}$")
ax2.set_title(R"|$\mathcal{L} - \mathcal{L}_\text{EM}$|")
cbar3 = fig.colorbar(pcm2, ax=ax2, shrink=0.7)
cbar3.ax.tick_params(labelsize=10)
plt.savefig("plots/heat_eval/matrix_heat.png")
plt.close()
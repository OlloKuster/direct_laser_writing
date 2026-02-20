import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/data/dlw/heat_sweep_adam/problems/metalens/plots/"

inds = np.linspace(0, 24, 25)
target_material = np.round(np.array([0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
target_void = np.round(np.array([0.8, 1., 1.2, 1.4, 1.6]) - 1, 1)
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

em_loss_matrix = np.clip(em_loss_matrix.reshape((5, 5)), 0, 100)
loss_matrix = np.clip(loss_matrix.reshape((5, 5)), 0, 100)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
pcm0 = ax0.imshow(em_loss_matrix.T, origin='lower', cmap='RdBu_r')
ax0.set_xlabel("Material")
ax0.set_ylabel("Void")
ax0.set_xticks([0, 1, 2, 3, 4])
ax0.set_yticks([0, 1, 2, 3, 4])
ax0.set_xticklabels(target_material)
ax0.set_yticklabels(target_void)
ax0.set_title(r"$v_\text{lens}$")
fig.colorbar(pcm0, ax=ax0, shrink=0.7)
pcm1 = ax1.imshow(loss_matrix.T, origin='lower', cmap='RdBu_r')
ax1.set_xlabel("Material")
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(target_material)
ax1.set_title("Total Loss")
fig.colorbar(pcm1, ax=ax1, shrink=0.7)
pcm2 = ax2.imshow(np.abs(loss_matrix - em_loss_matrix).T, origin='lower')
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(target_material)
ax2.set_xlabel("Material")
ax2.set_title("Difference")
fig.colorbar(pcm2, ax=ax2, shrink=0.7)
plt.show()
plt.savefig("plots/heat_eval/matrix.png")
plt.close()
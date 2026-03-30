import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path_feature_size = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_robust_sweep/"

lp_diff = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
inds_lp_diff = np.linspace(0, len(lp_diff)-1, len(lp_diff))
lp_diff_rel = np.array(lp_diff)
# i = 0

em_loss_eroded = np.zeros_like(lp_diff_rel)
em_loss_regular = np.zeros_like(lp_diff_rel)
em_loss_dilated = np.zeros_like(lp_diff_rel)
loss_list = np.zeros_like(lp_diff_rel)


for i in inds_lp_diff:
    with h5py.File(path_feature_size + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss = grp["em_loss"][:]
        loss = grp["loss"][:]
        em_loss_eroded[int(i)] = em_loss[-1][0]
        em_loss_regular[int(i)] = em_loss[-1][1]
        em_loss_dilated[int(i)] = em_loss[-1][2]
        loss_list[int(i)] = loss[-1]


fig, axs = plt.subplots(1, 1)
# plt.plot(em_loss_eroded)
# plt.plot(em_loss_regular)
# plt.plot(em_loss_dilated)
plt.plot(loss_list)
plt.show()

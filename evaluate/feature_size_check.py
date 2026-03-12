import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path_feature_size = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/feature_size_check/"

inds_feature_size = np.linspace(0, 6, 7)
feature_size_factor = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
feature_size = np.array(feature_size_factor) / np.sqrt(3)
# i = 0

em_loss_list_feature_size = np.zeros(7)


for i in inds_feature_size:
    with h5py.File(path_feature_size + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss_feature_size = grp["em_loss"][:]
        loss_feature_size = grp["loss"][:]
        em_loss_list_feature_size[int(i)] = np.max(em_loss_feature_size)

path_lp = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_sweep/"

inds_lp = np.linspace(0, 10, 11)
lps = np.linspace(0.001, 0.01, 11) / 0.0019 * 20
# i = 0

em_loss_list_lp = np.zeros(11)


for i in inds_lp:
    with h5py.File(path_lp + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss_lp = grp["em_loss"][:]
        loss_lp = grp["loss"][:]
        em_loss_list_lp[int(i)] = em_loss_lp[-1]


print(em_loss_list_lp)
fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
lns1 = ax1.plot(lps, em_loss_list_lp, color='black', label='Laser Power')
ax1.set_xlabel(r"Rel. Laser Power$\,$(%)", fontsize=18)
ax1.set_ylabel(r"$\mathcal{L}_\text{EM}$", fontsize=18)
ax1.set_xlim(lps[0], 100)
ax1.scatter(lps[3], em_loss_list_lp[3], c='black')

ax2 = ax1.twiny()
lns2 = ax2.plot(feature_size[::-1], em_loss_list_feature_size[::-1], color='gray', label='Min. Feature Size')
ax2.set_xlabel(r"Min. Feature Size$\,$($\mathregular{\mu}$m)", fontsize=18)
ax2.set_xlim(feature_size[-1], feature_size[0])
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4)
plt.savefig("plots/feature_size_comparison.png")
plt.close()
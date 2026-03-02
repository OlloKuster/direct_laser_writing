import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)

path = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/resolution_check_new/"

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


plt.plot(feature_size, em_loss_list)
plt.show()
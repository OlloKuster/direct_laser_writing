import h5py
import numpy as np
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader
from projection._projection_loader import projection_loader

with h5py.File("../data/feature_size_check/data_5_inf.h5") as f:
    grp = f["lens_3d"]
    rho_00 = grp["rho"][:]

gaussian_filter = filter_loader("gauss_jax", 2 / np.sqrt(3))
projection = projection_loader("tanh_jax", 0.5, np.inf, 14)

rho_0 = np.concatenate((rho_00, np.flip(rho_00, axis=0)), axis=0)
rho_0 = np.concatenate((rho_0, np.flip(rho_0, axis=1)), axis=1)

rho_init_bin_0 = projection(rho_00)
rho_init_bin = np.concatenate((rho_init_bin_0, np.flip(rho_init_bin_0, axis=0)), axis=0)
rho_init_bin = np.concatenate((rho_init_bin, np.flip(rho_init_bin, axis=1)), axis=1)

rho_filt_0 = gaussian_filter(rho_init_bin_0)


rho_filt = np.concatenate((rho_filt_0, np.flip(rho_filt_0, axis=0)), axis=0)
rho_filt = np.concatenate((rho_filt, np.flip(rho_filt, axis=1)), axis=1)

rho_bin = projection(rho_filt_0)
rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=0)), axis=0)
rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=1)), axis=1)

plt.figure(figsize=(6, 6))
plt.imshow(rho_init_bin[rho_init_bin.shape[0]//2-5].T, origin='lower', cmap='binary')
plt.axis('off')
plt.savefig("plots/gauss_filter/rho_gauss.png")
plt.close()
plt.figure(figsize=(6, 6))
plt.imshow(rho_filt[rho_filt.shape[0]//2-5].T, origin='lower', cmap='binary')
plt.axis('off')
plt.savefig("plots/gauss_filter/filter_gauss.png")
plt.close()
plt.figure(figsize=(6, 6))
plt.imshow(rho_bin[rho_bin.shape[0]//2-5].T, origin='lower', cmap='binary')
plt.axis('off')
plt.savefig("plots/gauss_filter/binary_gauss.png")
plt.close()
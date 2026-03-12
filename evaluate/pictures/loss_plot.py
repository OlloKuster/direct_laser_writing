import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

with h5py.File("../data/feature_size_check/data_5_inf.h5") as f:
    grp = f["lens_3d"]
    loss = grp["loss"][:]

font = {'family': 'sans-serif',
        'size': 30}
matplotlib.rc('font', **font)

plt.figure(figsize=(12, 10))
plt.plot(loss)
plt.ylabel(r"$\mathcal{L}(\rho)$")
plt.xlabel("Iteration")
plt.ylim(0, np.max(loss)+0.3)
plt.xlim(0, 30)
plt.tight_layout()
plt.show()

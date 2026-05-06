import pyvista as pv
import h5py
import numpy as np

base = h5py.File(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/power_splitter/plots/data_0_16.h5", "r")
grp = base["power_splitter"]
print(grp.keys())
eps = grp["eps_normal"][:]
base.close()

eps = eps[:, eps.shape[1] // 2:, eps.shape[2] // 2:]

resolution = 14



eps = np.concatenate((np.flip(eps, axis=1), eps), axis=1)
eps = np.concatenate((np.flip(eps, axis=2), eps), axis=2)

print(np.any(eps < 0) or np.any(eps > 1))

p = pv.Plotter(off_screen=False)
data = pv.wrap(np.array(eps))
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show()

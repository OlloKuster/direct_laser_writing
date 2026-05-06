import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

with h5py.File("../data/feature_size_check/data_7_inf.h5") as f:
        grp = f["lens_3d"]
        eps_nothing = grp["eps"][:]
        rho = grp["rho"][:]
        E_nothing = grp["E"][0]
        print(grp["loss"][-1])

# pv.plot(eps_nothing, cmap='binary'
print(rho.shape[2]//14)

eps_nothing = np.ones_like(eps_nothing) * 1.53**2

data_nothing = pv.wrap(eps_nothing)
data_e_nothing = pv.wrap(50*np.clip(np.abs(E_nothing), 7.5e-3, 100))

p = pv.Plotter(off_screen=True)
p.add_mesh(data_nothing.contour(), cmap='binary')
# p.add_volume(data_e_nothing, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/gauss.png')
p.close()

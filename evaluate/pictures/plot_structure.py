import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

with h5py.File("../data/feature_size_check/data_5_inf.h5") as f:
        grp = f["lens_3d"]
        eps_nothing = grp["eps"][:]
        E_nothing = grp["E"][0]
        print(grp["loss"][-1])

# pv.plot(eps_nothing, cmap='binary')

data_nothing = pv.wrap(eps_nothing)
data_e_nothing = pv.wrap(50*np.clip(np.abs(E_nothing), 7.5e-3, 100))

p = pv.Plotter()
p.add_mesh(data_nothing.contour(), cmap='binary')
p.add_volume(data_e_nothing, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/gauss.png')
p.close()


with h5py.File("../data/feature_size_check/data_dlw_inf.h5") as f:
        grp = f["lens_3d"]
        eps_dlw = grp["eps"][:]
        E_dlw = grp["E"][0]
        print(grp["loss"][-1])

# pv.plot(eps_nothing, cmap='binary')

data_dlw = pv.wrap(eps_dlw)
data_e_dlw = pv.wrap(50*np.clip(np.abs(E_dlw), 4.5e-3, 100))

p = pv.Plotter()
p.add_mesh(data_dlw.contour(), cmap='binary')
p.add_volume(data_e_dlw, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/dlw.png')
p.close()
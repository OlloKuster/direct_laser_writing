import h5py
import pyvista as pv
import numpy as np

with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/pictures/plots/data_0_8.h5") as f:
    grp = f["lens_3d"]
    print(grp.keys())
    eps_nothing = grp["eps_normal"][:]
    E_nothing = grp["E_normal"][0]
    print(grp["loss"][-1])

data_nothing = pv.wrap(eps_nothing)
data_e_nothing = pv.wrap(50*np.clip(np.abs(E_nothing), 6e-3, 100))

p = pv.Plotter(off_screen=True)
p.add_mesh(data_nothing.contour(), cmap='binary')
p.add_volume(data_e_nothing, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.1)
p.show(screenshot='plots/big_lens.png')
p.close()



with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/polarization_splitter/plots/data_0_16.h5") as f:
    grp = f["polarization_splitter"]
    print(grp.keys())
    eps_nothing = grp["eps"][:]
    E = grp["E"]

print(E)
import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/simulation/big_metalens.h5") as f:
        grp = f["rho"]
        rho = grp["rho"][:]
pv.global_theme.allow_empty_mesh = True

data_nothing = pv.wrap(rho)

p = pv.Plotter(off_screen=False)
p.add_mesh(data_nothing.contour(), cmap='binary')
# p.add_volume(data_e_nothing, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/metalens_init.png')
p.close()

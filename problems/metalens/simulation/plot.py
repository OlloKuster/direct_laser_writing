import pyvista as pv
import h5py
import matplotlib.pyplot as plt
import numpy as np


base = h5py.File(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/data.h5", "r")
grp = base["lens_3d"]
eps = grp["eps"][:]
rho_0 = grp["rho"][:]
base.close()

p = pv.Plotter()
data = pv.wrap(eps)
p.add_mesh(data.contour(), cmap='binary')
p.show()


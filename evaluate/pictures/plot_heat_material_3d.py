import h5py
import numpy as np
import torch
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv

from filtering._filter_loader import filter_loader
from projection._projection_loader import projection_loader
from tofea.fea2d import FEA2D_T
from tofea.fea3d import FEA3D_T
from utility.helper import f2param

with h5py.File("../data/feature_size_check/data_5_inf.h5") as f:
    grp = f["lens_3d"]
    rho_00 = grp["rho"][:]

# rho_00 = np.ones_like(rho_00)
# rho_00[:, :rho_00.shape[1]//2] = 0
gaussian_filter = filter_loader("gauss_jax", 2 / np.sqrt(3))
dlw_filter = filter_loader("dose_conv", 14, 2 * 0.0021)
projection = projection_loader("tanh_jax", 0.5, np.inf, 14)

rho_gauss_filt_0 = gaussian_filter(np.array(rho_00, dtype=np.float64))
rho_bin = projection(rho_gauss_filt_0)
rho_bin_heat = rho_bin

heat_sinks_matter = jnp.zeros((rho_bin.shape[0] + 1,
                               rho_bin.shape[1] + 1,
                               rho_bin.shape[2] + 1), dtype='?')
heat_sinks_matter = heat_sinks_matter.at[..., 0].set(True)
kappa_r_matter = f2param(rho_bin_heat, (1e-5, 1))
fem_matter = FEA3D_T(heat_sinks_matter)
src_matter = jnp.pad(rho_bin_heat, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
T_matter = fem_matter.temperature(kappa_r_matter, src_matter).reshape(heat_sinks_matter.shape)
T_matter = T_matter[:-1, :-1, :-1]

rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=0)), axis=0)
rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=1)), axis=1)


T_matter = np.concatenate((T_matter, np.flip(T_matter, axis=0)), axis=0)
T_matter = np.concatenate((T_matter, np.flip(T_matter, axis=1)), axis=1)

print(np.max(T_matter))
T_max = np.max(T_matter)
p = pv.Plotter(off_screen=True)
data_gauss = pv.wrap(rho_bin)
p.add_mesh(data_gauss.contour(), cmap='binary')
p.add_volume(T_matter, opacity=0.1, cmap='hot')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.2)
p.show(screenshot="plots/heat_gauss.png")
p.close()

p = pv.Plotter(off_screen=False)
data_gauss = pv.wrap(rho_bin)
p.add_mesh(data_gauss.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.2)
p.show(screenshot="plots/heat_gauss_density.png")
p.close()



with h5py.File("../data/feature_size_check/data_dlw_inf.h5") as f:
    grp = f["lens_3d"]
    rho_00 = grp["rho"][:]

rho_init_bin_0 = projection(rho_00)
rho_filt_0 = dlw_filter(
    torch.tensor(np.array(rho_init_bin_0, dtype=np.float64), device='cuda')).detach().cpu().numpy()
rho_bin = projection(rho_filt_0)
rho_bin_heat = rho_bin

heat_sinks_matter = jnp.zeros((rho_bin.shape[0] + 1,
                               rho_bin.shape[1] + 1,
                               rho_bin.shape[2] + 1), dtype='?')
heat_sinks_matter = heat_sinks_matter.at[..., 0].set(True)
kappa_r_matter = f2param(rho_bin_heat, (1e-5, 1))
fem_matter = FEA3D_T(heat_sinks_matter)
src_matter = jnp.pad(rho_bin_heat, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
T_matter = fem_matter.temperature(kappa_r_matter, src_matter).reshape(heat_sinks_matter.shape)
T_matter = T_matter[:-1, :-1, :-1]
rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=0)), axis=0)
rho_bin = np.concatenate((rho_bin, np.flip(rho_bin, axis=1)), axis=1)

print(np.max(T_matter))

T_matter = np.concatenate((T_matter, np.flip(T_matter, axis=0)), axis=0)
T_matter = np.concatenate((T_matter, np.flip(T_matter, axis=1)), axis=1)
T_matter[-1, -1, -1] = T_max

p = pv.Plotter(off_screen=True)
data_gauss = pv.wrap(rho_bin)
p.add_mesh(data_gauss.contour(), cmap='binary')
p.add_volume(T_matter, opacity=0.1, cmap='hot')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot="plots/heat_dlw.png")
p.close()
p = pv.Plotter(off_screen=True)
data_gauss = pv.wrap(rho_bin)
p.add_mesh(data_gauss.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 30
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot="plots/heat_dlw_density.png")
p.close()


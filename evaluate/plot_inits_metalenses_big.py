import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm

from problems.metalens.simulation.config_structure import ConfigSim
from utility.helper import f2param, split_int

with h5py.File("/scratch/local/okuster/data/dlw/big_metalens/fy4301-dlw_plots/dlw_params_paper_more_heat/problems/metalens/plots/data_0_inf.h5") as f:
        grp = f["lens_3d"]
        rho = grp["rho"][:]
        eps = grp["eps_normal"][:]
        E = grp["E_normal"][0]


pv.global_theme.allow_empty_mesh = True

data = pv.wrap(eps)
data_e = pv.wrap(100*np.clip(np.abs(E), 7.5e-3, 100))


p = pv.Plotter(off_screen=True)
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/metalens_big_final.png')
p.close()

p = pv.Plotter(off_screen=True)
p.add_mesh(data.contour(), cmap='binary')
p.add_volume(data_e, cmap='RdBu')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/metalens_big_final_field.png')
p.close()

import jax.numpy as jnp

resolution = 14
mask = np.ones_like(rho)
mask[:int(ConfigSim.buffer_side * resolution)] = 0
mask[:, :int(ConfigSim.buffer_side * resolution)] = 0
mask[:, :, -int(ConfigSim.buffer_top * resolution):] = 0

def preprocess(rho, resolution):
    rho_p = rho.at[:, :, :int(jnp.ceil(0.5 * resolution))].set(1)
    return rho_p

rho = jnp.array(np.round(rho*mask))

simulation_domain = (int(jnp.ceil(ConfigSim.simulation_domain_shape[0] * resolution)),
                     int(jnp.ceil(ConfigSim.simulation_domain_shape[1] * resolution)),
                     int(jnp.ceil(ConfigSim.simulation_domain_shape[2] * resolution)))
size_rho = (int(jnp.ceil(2 * ConfigSim.rho_shape[0] * resolution)),
            int(jnp.ceil(2 * ConfigSim.rho_shape[1] * resolution)),
            int(jnp.ceil(ConfigSim.rho_shape[2] * resolution)))

rho_p = preprocess(rho[:-ConfigSim.buffer_side * resolution, :-ConfigSim.buffer_side * resolution], resolution)

eps = f2param(rho_p, ConfigSim.epsilon)

eps = jnp.concatenate((eps, jnp.flip(eps, axis=0)), axis=0)
eps = jnp.concatenate((eps, jnp.flip(eps, axis=1)), axis=1)

eps = jnp.pad(eps,
              [split_int(simulation_domain[0] - size_rho[0])] +
              [split_int(simulation_domain[1] - size_rho[1])] +
              [(0, int(jnp.ceil((ConfigSim.space_top + ConfigSim.dpml) * resolution)))], mode='constant',
              constant_values=ConfigSim.epsilon[0])

eps = jnp.pad(eps,
              [(0, 0)] * 2 + [(int(jnp.ceil((ConfigSim.buffer_bottom + ConfigSim.dpml) * resolution)), 0)],
              mode='constant',
              constant_values=ConfigSim.epsilon[1])

import matplotlib.pyplot as plt
plt.imshow(eps[:, :, eps.shape[2]//2].T, origin='lower', cmap='binary')
plt.show()




data = pv.wrap(np.array(eps))
p = pv.Plotter(off_screen=False)
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'yz'
p.camera.elevation = 20
p.camera.azimuth = - 45
p.remove_scalar_bar()
p.camera.zoom(1.2)
p.show(screenshot='plots/init_structures/metalens_big_rho.png')
p.close()



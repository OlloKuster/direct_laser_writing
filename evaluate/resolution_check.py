import jax.numpy as jnp
import jax
import numpy as np
import h5py
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader

with h5py.File("data/data_4_32.h5") as f:
    grp = f["lens_3d"]
    rho_0 = grp["rho"][:]
    f.close()
foms = []
resolutions = [2, 4, 6, 8, 10, 12, 14]

for resolution in resolutions:
    resize_shape = (ConfigSim.rho_shape[0] * resolution, ConfigSim.rho_shape[1] * resolution, ConfigSim.rho_shape[2] * resolution)
    filter = filter_loader("gauss_jax", 1 / (4*np.sqrt(3)) * resolution)
    projection = projection_loader("ssp_jax", 0.5, 32, resolution)

    rho_0 = jax.image.resize(rho_0, resize_shape, 'bicubic', antialias=False)

    # rho_proj_init = rho_0 * mask

    # rho_0 = convert_to(rho_proj_init, conversions)

    rho_opt_filtered = filter(rho_0)

    # rho_opt_filtered = convert_to(rho_opt_filtered, backconversions)

    rho_opt_proj = projection(jnp.array(rho_opt_filtered))

    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)
    currents = jnp.ones(size_currents, jnp.complex128)


    E, eps = em_simulation(jnp.array(rho_opt_proj), currents, resolution)

    focal_spot = E[0][
                E[0].shape[0] // 2, E[0].shape[1] // 2, int(jnp.ceil(ConfigSim.location_focal_spot * resolution))]

    print(np.abs(focal_spot))

    foms.append(np.abs(focal_spot))

fom_0 = foms[-3]
diffs = np.abs(np.array(foms) - fom_0) / fom_0
plt.plot(resolutions, diffs)
plt.show()

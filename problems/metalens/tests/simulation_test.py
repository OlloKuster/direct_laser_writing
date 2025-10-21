import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import time

from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation


def test(seed):
    jax.config.update("jax_enable_x64", True)
    resolution = 10
    if seed == 0:
        rho_0 = np.ones((ConfigSim.rho_shape[0]*resolution,
                        ConfigSim.rho_shape[1]*resolution,
                        ConfigSim.rho_shape[2]*resolution)) * 0.5

    currents = jnp.ones((ConfigSim.currents_shape[0]*resolution, ConfigSim.currents_shape[1]*resolution, 1),
                       jnp.complex128)

    start = time.time()

    E, eps = em_simulation(rho_0, currents, resolution)

    print(time.time() - start)

    plt.imshow(eps[eps[0].shape[0]//2].T, origin='lower', cmap='binary',
               extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
    plt.imshow(np.abs(E[0])[E[0].shape[0]//2].T, origin='lower', cmap='magma',
               extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]), alpha=0.8)
    plt.show()

if __name__ == '__main__':
    test(0)




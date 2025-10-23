import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import meep as mp

import time

from mode_calculator.meep_eigenmode_calculator_3D import find_mode_profile
from problems.mode_converter_jax.simulation.config_structure import ConfigSim
from problems.mode_converter_jax.simulation.simulation import em_simulation


def test(seed):
    jax.config.update("jax_enable_x64", True)
    resolution = 10
    if seed == 0:
        rho_0 = np.ones((ConfigSim.rho_shape[0]*resolution,
                        ConfigSim.rho_shape[1]*resolution,
                        ConfigSim.rho_shape[2]*resolution))

    currents = find_mode_profile(ConfigSim.simulation_domain_shape, resolution,
                                    (ConfigSim.wg_width, ConfigSim.wg_width),
                                    ConfigSim.epsilon, ConfigSim.wavelength,
                                    d_sub=ConfigSim.buffer_sub + ConfigSim.dpml,
                                    mode=1, parity=mp.EVEN_Y, field=mp.Ez)

    currents = np.reshape(currents, (1, currents.shape[0], currents.shape[1]))

    start = time.time()

    E, eps = em_simulation(rho_0, currents, resolution)

    print(time.time() - start)

    pos_y = int(jnp.ceil((ConfigSim.dpml+ConfigSim.buffer_sub+ConfigSim.wg_width/2) * resolution))

    plt.imshow(eps[:, pos_y].T, origin='lower', cmap='binary',
               )
    plt.imshow(np.abs(E[2])[:, pos_y].T, origin='lower', cmap='magma',
               alpha=0.8)
    plt.show()

    plt.imshow(eps[:, :, eps.shape[2]//2].T, origin='lower', cmap='binary',
               )
    plt.imshow(np.abs(E[2])[:, :, E[2].shape[2]//2].T, origin='lower', cmap='magma',
               alpha=0.8)
    plt.show()


if __name__ == '__main__':
    test(0)




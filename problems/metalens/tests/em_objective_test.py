import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import time

from problems.metalens._objective_loader import objective_loader
from problems.metalens.config_structure import ConfigSim
from problems.metalens.simulation import em_simulation


def test(seed):
    jax.config.update("jax_enable_x64", True)
    resolution = 6
    if seed == 0:
        rho_0 = np.ones((ConfigSim.rho_shape[0]*resolution,
                        ConfigSim.rho_shape[1]*resolution,
                        ConfigSim.rho_shape[2]*resolution)) * 0.5

    currents = jnp.ones((ConfigSim.currents_shape[0]*resolution, ConfigSim.currents_shape[1]*resolution, 1),
                       jnp.complex128)

    objective_em = objective_loader("em_only", currents, resolution, 1)
    fom = objective_em(rho_0)
    print(fom)

if __name__ == "__main__":
    test(0)
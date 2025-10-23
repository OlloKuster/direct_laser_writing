from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ConfigSim:
    """
    Config the 3D lens, defines the required parameters which are used for the simulation.
    """
    rho_shape = (6, 4, 4)
    buffer_side = 1
    buffer_top = 1
    wg_length = 3
    wg_width = 4
    buffer_sub = 1
    dpml = 0.5
    simulation_domain_shape = (int(jnp.ceil(dpml + wg_length + rho_shape[0] + wg_length + dpml)),
                               int(jnp.ceil(dpml + buffer_sub + rho_shape[1] + buffer_top + dpml)),
                               int(jnp.ceil(dpml + buffer_side + rho_shape[2] + buffer_side + dpml)))

    currents_shape = (simulation_domain_shape[1], simulation_domain_shape[2])

    wavelength = 1.

    location_currents = dpml + 1

    epsilon = (1, 1.444**2, 1.53**2)

    monitor_pos = (dpml + wg_length + rho_shape[0] + 0.1,
                   dpml + buffer_sub + wg_width / 2,
                   simulation_domain_shape[2] / 2)

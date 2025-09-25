from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ConfigSim:
    """
    Config the 3D lens, defines the required parameters which are used for the simulation.
    """
    rho_shape = (10, 10, 5)
    buffer_side = 1
    buffer_top = 3
    buffer_bottom = 1
    dpml = 0.5
    simulation_domain_shape = (int(jnp.ceil(dpml + buffer_side + 2*rho_shape[0] + buffer_side + dpml)),
                               int(jnp.ceil(dpml + buffer_side + 2*rho_shape[1] + buffer_side + dpml)),
                               int(jnp.ceil(dpml + buffer_bottom + rho_shape[2] + buffer_top + dpml)))
    currents_shape = (simulation_domain_shape[0], simulation_domain_shape[1])
    sinks_shape = rho_shape
    wavelength = 1.55
    save_plot = True
    save_data = True
    epsilon = (1, 2.25)
    kappa = (1e-5, 1)
    resize_factor = 1

    location_currents = dpml + 1
    location_focal_spot = 8

    TARGET_EM = 20
    TARGET_MATERIAL = -1
    TARGET_VOID = 1.5

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class ConfigSim:
    """
    Config the 3D lens, defines the required parameters which are used for the simulation.
    """
    rho_shape = (5, 5, 4)
    buffer_side = 1  # Buffer for the mask.
    buffer_top = 1.  # Buffer for the mask.
    space_top = 3  # Actual space above the design region
    buffer_bottom = 1  # Thickness "substrate"
    dpml = 0.5  # Thickness PML
    simulation_domain_shape = (int(jnp.ceil(dpml + 2 * rho_shape[0] + dpml)),
                               int(jnp.ceil(dpml + 2 * rho_shape[1] + dpml)),
                               int(jnp.ceil(dpml + buffer_bottom + rho_shape[2] + space_top + dpml)))
    currents_shape = (simulation_domain_shape[0], simulation_domain_shape[1])
    sinks_shape = rho_shape
    wavelength = 1.55
    epsilon = (1, 1.53**2)  # Permittivity.
    kappa = (1e-5, 1)  # Thermal conductivity
    resize_factor = 1  # Scales down the thermal simulation in case the FEM-mesh takes too much memory.

    location_currents = dpml + 1
    location_focal_spot = dpml + buffer_bottom + rho_shape[2] + buffer_top + 0.25

    TARGET_EM = 30  # Target EM Performance. Given in field enhancement at the focal spot
    TARGET_MATERIAL = 1.6  # Normalizes the target heat_eval for the material. The initial value is 1-TARGET_MATERIAL.
    TARGET_VOID = 1.6  # Normalizes the target heat_eval for the void. The initial value is 1-TARGET_VOID.

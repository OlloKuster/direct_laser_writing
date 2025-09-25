import jax.numpy as jnp

from problems.metalens.config_structure import ConfigSim
from problems.metalens.simulation import em_simulation


def objective_em_f(currents, resolution, init_value):
    def objective_em(rho):
        E, eps = em_simulation(rho, currents, resolution)
        focal_spot = E[0][E[0].shape[0] // 2, E[0].shape[1] // 2, ConfigSim.location_focal_spot*resolution]
        return jnp.abs(focal_spot) / init_value
    return objective_em

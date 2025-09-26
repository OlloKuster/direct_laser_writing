import jax.numpy as jnp

from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation, heat_simulation
from utility.helper import softplus


def objective_em_f(currents, resolution, init_value):
    def objective_em(rho):
        E, eps = em_simulation(rho, currents, resolution)
        focal_spot = E[0][E[0].shape[0] // 2, E[0].shape[1] // 2, ConfigSim.location_focal_spot * resolution]
        return jnp.abs(focal_spot) / init_value

    return objective_em


def objective_heat_f():
    def objective_heat(rho):
        T_mat, T_void, _ = heat_simulation(rho, ConfigSim.resize_factor)
        return T_mat, T_void

    return objective_heat


def objective_em_heat_f(currents, resolution, init_values):
    def objective_em(rho):
        E, eps = em_simulation(rho, currents, resolution)
        focal_spot = E[0][E[0].shape[0] // 2, E[0].shape[1] // 2, ConfigSim.location_focal_spot * resolution]
        return jnp.abs(focal_spot) / init_values[0]

    def objective_heat(rho):
        T_mat, T_void, _ = heat_simulation(rho, ConfigSim.resize_factor)
        return T_mat, T_void

    def objective_softplus(rho):
        v_lens = objective_em(rho)
        v_heat_m, v_heat_v = objective_heat(rho)

        n_lens = (ConfigSim.TARGET_EM - v_lens) / ConfigSim.TARGET_EM
        n_heat_m = (v_heat_m - init_values[1]) / init_values[1]
        n_heat_v = (v_heat_v - init_values[2]) / init_values[2]

        objs = jnp.array([n_lens, n_heat_m, n_heat_v])

        return jnp.linalg.norm(softplus(objs))

    return objective_softplus

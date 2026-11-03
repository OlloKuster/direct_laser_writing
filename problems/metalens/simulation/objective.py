import jax.numpy as jnp
import jax.nn as nn
from jax.debug import print as jprint

from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import printed, em_simulation, heat_simulation
from utility.helper import relu


def soft_min(foms):
    """
    Soft minimum over the process variants, so that the worst performing one is optimized for. For a single variant
    it is exactly its figure of merit.
    :param foms: Figures of merit of all process variants [V].
    :return: Soft minimum of foms.
    """
    return -nn.logsumexp(-foms)


def focal_spot(E, resolution, init_value):
    """
    :param E: Electric field.
    :param resolution: Resolution of the simulation [px/um].
    :param init_value: Initial value of the problem to make a unitless comparison possible.
    :return: Field enhancement at the focal spot of the x-component of E.
    """
    E_focal = E[0][E[0].shape[0] // 2, E[0].shape[1] // 2, int(jnp.ceil(ConfigSim.location_focal_spot * resolution))]
    return jnp.abs(E_focal) / init_value


def _log(logs):
    print("====================")
    for name, log in logs.items():
        jprint("{name}:", name=name)
        jprint("    {log}", log=log)
    print("====================")


def objective_em_f(init_projection, filter, projection, currents, resolution, init_value):
    """
    Generates the pure EM-objective function.
    :param currents: Source currents of the simulation.
    :param resolution: Resolution of the simulation [px/um].
    :param init_value: Initial value of the problem to make a unitless comparison possible.
    :return: Pure EM-Objective function.
    """

    def objective_em(rho):
        """
        Pure EM-objective function, which evaluates the field enhancement at the focal spot of every process variant.
        :param rho: Density (design variable) of the problem [0, 1].
        :return: (Soft minimum of the field enhancements, (field enhancements [V], eps of the nominal structure)).
        """
        v_lens, eps = [], []
        for rho_p in printed(rho, init_projection, filter, projection):
            E, eps_p = em_simulation(rho_p, currents, resolution)
            v_lens.append(focal_spot(E, resolution, init_value))
            eps.append(eps_p)
        v_lens = jnp.stack(v_lens)
        return soft_min(v_lens), (v_lens, eps[len(eps) // 2])

    return objective_em


def objective_em_heat_f(init_projection, filter, projection, currents, resolution, init_values):
    """
    Generates the (softmax) connectivity-objective function. With the robust filter, the worst performing of the
    eroded, normal and dilated structure is optimized for.
    :param currents: Source currents of the simulation.
    :param resolution: Resolution of the simulation [px/um].
    :param init_values: Initial value of the problem to make a unitless comparison possible.
                        Tuple of initial values for the EM, material heat_eval and void heat_eval simulation.
    :return: Connectivity-objective funtion.
    """

    def objective_softplus(rho_p):
        """
        Generates the softplus objective function for one printed structure. Implements the virtual heat_eval method
        and filters it through a non-linear activation function. This is to make sure that once connectivity is
        reached, the optimizer disregards the contributions of the heat_eval solver.
        :param rho_p: Printed density [0, 1].
        :return: (Softplus objective function, field enhancement, permittivity).
        """
        E, eps = em_simulation(rho_p, currents, resolution)
        v_lens = focal_spot(E, resolution, init_values[0])
        v_heat_m, v_heat_v, _ = heat_simulation(rho_p)

        n_heat_m = (v_heat_m - init_values[1]) / init_values[1]
        n_heat_v = (v_heat_v - init_values[2]) / init_values[2]

        _log({
            "v_lens": v_lens, "n_heat_m": n_heat_m, "n_heat_v": n_heat_v, "v_heat_m": relu(n_heat_m),
            "v_heat_v": relu(n_heat_v)
        })

        objs = jnp.array([n_heat_m, n_heat_v])

        return v_lens * (1 - jnp.linalg.norm(relu(objs))), v_lens, eps

    def objective_robust_softplus(rho):
        """
        Evaluates the softplus objective function for every process variant and takes the soft minimum.
        :param rho: Density (design variable) of the problem [0, 1].
        :return: (Soft minimum of the figures of merit, (field enhancements [V], eps of the nominal structure)).
        """
        foms, v_lens, eps = zip(*(objective_softplus(rho_p)
                                  for rho_p in printed(rho, init_projection, filter, projection)))
        foms, v_lens = jnp.stack(foms), jnp.stack(v_lens)
        fom_min = soft_min(foms)

        _log({"v_lens": v_lens, "fom": foms, "fom_min": fom_min})

        return fom_min, (v_lens, eps[len(eps) // 2])

    return objective_robust_softplus

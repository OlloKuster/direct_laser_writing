import jax.numpy as jnp
import jaxwell

from problems.metalens.simulation.config_structure import ConfigSim
from utility.helper import f2param, split_int


def em_simulation(rho, currents, resolution):
    simulation_domain = (ConfigSim.simulation_domain_shape[0] * resolution,
                         ConfigSim.simulation_domain_shape[1] * resolution,
                         ConfigSim.simulation_domain_shape[2] * resolution)
    size_rho = (int(jnp.ceil(2 * ConfigSim.rho_shape[0] * resolution)),
                int(jnp.ceil(2 * ConfigSim.rho_shape[1] * resolution)),
                int(jnp.ceil(ConfigSim.rho_shape[2] * resolution)))

    omega = 2 * jnp.pi / (ConfigSim.wavelength * resolution)

    eps = f2param(rho, ConfigSim.epsilon)

    eps = jnp.concatenate((eps, jnp.flip(eps, axis=0)), axis=0)
    eps = jnp.concatenate((eps, jnp.flip(eps, axis=1)), axis=1)

    eps = jnp.pad(eps,
                  [split_int(simulation_domain[0] - size_rho[0])] +
                  [split_int(simulation_domain[1] - size_rho[1])] +
                  [(0, int(jnp.ceil((ConfigSim.buffer_top+ConfigSim.dpml)*resolution)))], mode='constant',
                  constant_values=ConfigSim.epsilon[0])



    eps = jnp.pad(eps,
                  [(0, 0)] * 2 + [(int(jnp.ceil((ConfigSim.buffer_bottom+ConfigSim.dpml)*resolution)), 0)],
                  mode='constant',
                  constant_values=ConfigSim.epsilon[1])


    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)

    currents = currents / jnp.linalg.norm(currents)

    b = jnp.pad(currents,
                [split_int(simulation_domain[0] - size_currents[0])] +
                [split_int(simulation_domain[1] - size_currents[1])] +
                [(int(jnp.ceil(ConfigSim.location_currents * resolution - 1)),
                  int(jnp.ceil(simulation_domain[2] - ConfigSim.location_currents * resolution)))])
    b_zero = jnp.zeros(simulation_domain, jnp.complex128)

    eps_r = (eps, eps, eps)
    source_fields = (b, b_zero, b_zero)

    z = tuple(omega ** 2 * t for t in eps_r)
    b = tuple(jnp.complex128(-1j * omega * b) for b in source_fields)

    dpml = ConfigSim.dpml * resolution

    params = jaxwell.Params(
        pml_ths=((dpml, dpml), (dpml, dpml), (dpml, dpml)),
        pml_omega=omega,
        eps=1e-6,
        max_iters=1000000
    )

    E, _ = jaxwell.solve(params, z, b)

    return E, eps


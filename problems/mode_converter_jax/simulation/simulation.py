import jax.numpy as jnp
import jaxwell
from autograd import numpy as npa

from problems.mode_converter_jax.simulation.config_structure import ConfigSim
from utility.helper import f2param, split_int


def em_simulation(rho, currents, resolution):
    simulation_domain = (int(jnp.ceil(ConfigSim.simulation_domain_shape[0] * resolution)),
                         int(jnp.ceil(ConfigSim.simulation_domain_shape[1] * resolution)),
                         int(jnp.ceil(ConfigSim.simulation_domain_shape[2] * resolution)))
    size_rho = (int(jnp.ceil(ConfigSim.rho_shape[0] * resolution)),
                int(jnp.ceil(ConfigSim.rho_shape[1] * resolution)),
                int(jnp.ceil(ConfigSim.rho_shape[2] * resolution)))

    buffer_sub = int(jnp.ceil((ConfigSim.dpml + ConfigSim.buffer_sub) * resolution))
    buffer_top = int(jnp.ceil((ConfigSim.dpml + ConfigSim.buffer_top) * resolution))

    wg_width = int(jnp.ceil(ConfigSim.wg_width * resolution))

    omega = 2 * jnp.pi / (ConfigSim.wavelength * resolution)

    eps = f2param(rho, (ConfigSim.epsilon[0], ConfigSim.epsilon[2]))

    eps = jnp.pad(eps,
                  [split_int(simulation_domain[0] - size_rho[0])] + [
                      (0, buffer_top)] + [split_int(simulation_domain[2] - size_rho[2])],
                  mode='constant',
                  constant_values=ConfigSim.epsilon[0])

    eps = jnp.pad(eps,
                  [(0, 0)] + [(buffer_sub, 0)] + [(0, 0)],
                  mode='constant',
                  constant_values=ConfigSim.epsilon[1]
                  )
    eps = eps.at[-(simulation_domain[0] - size_rho[0]) // 2:,
          buffer_sub:buffer_sub + wg_width,
          (simulation_domain[2] - wg_width) // 2:(simulation_domain[2] + wg_width) // 2].set(ConfigSim.epsilon[2])
    eps = eps.at[:(simulation_domain[0] - size_rho[0]) // 2,
          buffer_sub:buffer_sub + wg_width,
          (simulation_domain[2] - wg_width) // 2:(simulation_domain[2] + wg_width) // 2].set(ConfigSim.epsilon[2])

    currents = currents / jnp.linalg.norm(currents)

    b = jnp.pad(currents,
                [(int(jnp.ceil(ConfigSim.location_currents * resolution - 1)),
                  int(simulation_domain[0] - ConfigSim.location_currents * resolution))] + [(0, 0)] * 2)
    b_zero = jnp.zeros(simulation_domain, jnp.complex128)

    eps_r = (eps, eps, eps)
    source_fields = (b_zero, b_zero, b)

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

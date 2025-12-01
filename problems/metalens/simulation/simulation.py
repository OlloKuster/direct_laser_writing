import jax
import jax.numpy as jnp
import jaxwell

from problems.metalens.simulation.config_structure import ConfigSim
from tofea.fea3d import FEA3D_T
from utility.helper import f2param, split_int


def em_simulation(rho, currents, resolution):
    # rho = rho.at[:, :, :resolution].set(1)
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


def heat_simulation(rho, resize_factor):
    rho_n_shape = (rho.shape[0] // resize_factor, rho.shape[1] // resize_factor, rho.shape[2] // resize_factor)
    rho_n = jax.image.resize(rho, rho_n_shape, 'bicubic', antialias=False)

    heat_sinks_matter = jnp.zeros((rho_n_shape[0] + 1,
                                   rho_n_shape[1] + 1,
                                   rho_n_shape[2] + 1), dtype='?')
    heat_sinks_matter = heat_sinks_matter.at[..., 0].set(True)
    kappa_r_matter = f2param(rho_n, ConfigSim.kappa)
    fem_matter = FEA3D_T(heat_sinks_matter)
    src_matter = jnp.pad(rho_n, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_matter = fem_matter.temperature(kappa_r_matter, src_matter)

    heat_sinks_void = jnp.zeros_like(heat_sinks_matter)
    heat_sinks_void = heat_sinks_void.at[0].set(True)
    heat_sinks_void = heat_sinks_void.at[-1].set(True)
    heat_sinks_void = heat_sinks_void.at[:, 0].set(True)
    heat_sinks_void = heat_sinks_void.at[:, -1].set(True)
    heat_sinks_void = heat_sinks_void.at[..., -1].set(True)
    kappa_r_void = f2param(1 - rho_n, ConfigSim.kappa)
    fem_void = FEA3D_T(heat_sinks_void)
    src_void = jnp.pad(1 - rho_n, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_void = fem_void.temperature(kappa_r_void, src_void)

    return jnp.sum(T_matter)/T_matter.size, jnp.sum(T_void)/T_void.size, kappa_r_matter

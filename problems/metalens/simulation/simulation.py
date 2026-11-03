import jax
import jax.numpy as jnp
import jaxwell

from problems.metalens.simulation.config_structure import ConfigSim
from tofea.fea3d import FEA3D_T
from utility.helper import f2param, split_int


def preprocess(rho, resolution):
    """
    Crops the side buffer (located at the lens center after mirroring) and sets the bottom 0.5um to material, so the
    lens is attached to the substrate.
    :param rho: Projected density including the side buffer.
    :param resolution: Resolution of the simulation [px/um].
    :return: Density of the size rho_shape * resolution.
    """
    buffer = int(ConfigSim.buffer_side * resolution)
    rho_p = jnp.asarray(rho)[:-buffer, :-buffer]
    rho_p = rho_p.at[:, :, :int(jnp.ceil(0.5 * resolution))].set(1)
    return rho_p


def _simulation_domain(resolution):
    """
    :param resolution: Resolution of the simulation [px/um].
    :return: Shape of the entire simulation domain [px].
    """
    return (int(jnp.ceil(ConfigSim.simulation_domain_shape[0] * resolution)),
            int(jnp.ceil(ConfigSim.simulation_domain_shape[1] * resolution)),
            int(jnp.ceil(ConfigSim.simulation_domain_shape[2] * resolution)))


def build_eps(rho_p, resolution):
    """
    Builds the permittivity of the entire simulation from a (printed) density. The density is cropped, mirrored and
    padded with the substrate and the space above the lens.
    :param rho_p: Printed density including the side buffer [0, 1].
    :param resolution: Resolution of the simulation [px/um].
    :return: Permittivity of the entire simulation.
    """

    simulation_domain = _simulation_domain(resolution)
    size_rho = (int(jnp.ceil(2 * ConfigSim.rho_shape[0] * resolution)),
                int(jnp.ceil(2 * ConfigSim.rho_shape[1] * resolution)),
                int(jnp.ceil(ConfigSim.rho_shape[2] * resolution)))

    eps = f2param(preprocess(rho_p, resolution), ConfigSim.epsilon)

    eps = jnp.concatenate((eps, jnp.flip(eps, axis=0)), axis=0)
    eps = jnp.concatenate((eps, jnp.flip(eps, axis=1)), axis=1)

    eps = jnp.pad(eps,
                  [split_int(simulation_domain[0] - size_rho[0])] +
                  [split_int(simulation_domain[1] - size_rho[1])] +
                  [(0, int(jnp.ceil((ConfigSim.space_top + ConfigSim.dpml) * resolution)))], mode='constant',
                  constant_values=ConfigSim.epsilon[0])

    eps = jnp.pad(eps,
                  [(0, 0)] * 2 + [(int(jnp.ceil((ConfigSim.buffer_bottom + ConfigSim.dpml) * resolution)), 0)],
                  mode='constant',
                  constant_values=ConfigSim.epsilon[1])

    return eps


def solve_em(eps, currents, resolution):
    """
    Solves the metalens problem for a given permittivity and source currents.
    :param eps: Permittivity of the entire simulation, e.g. from build_eps.
    :param currents: Source currents of the simulation.
    :param resolution: Resolution of the simulation [px/um].
    :return: Electric field.
    """

    simulation_domain = _simulation_domain(resolution)
    omega = 2 * jnp.pi / (ConfigSim.wavelength * resolution)

    size_currents = (int(jnp.ceil(ConfigSim.currents_shape[0] * resolution)),
                     int(jnp.ceil(ConfigSim.currents_shape[1] * resolution)),
                     1)

    currents = currents / jnp.linalg.norm(currents)

    b = jnp.pad(currents,
                [split_int(simulation_domain[0] - size_currents[0])] +
                [split_int(simulation_domain[1] - size_currents[1])] +
                [(int(jnp.ceil(ConfigSim.location_currents * resolution - 1)),
                  simulation_domain[2] - int(jnp.ceil(ConfigSim.location_currents * resolution)))])
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

    return E


def em_simulation(rho, init_proj, filter, proj, currents, resolution):
    """
    Simulates the metalens problem given a density and source currents.
    :param rho: Input density of the problem (design variable) [0, 1].
    :param currents: Source currents of the simulation.
    :param resolution: Resolution of the simulation [px/um].
    :return: (Electric field, permittivity of the entire simulation).
    """
    eps = build_eps(proj(filter(init_proj(rho))), resolution)
    return solve_em(eps, currents, resolution), eps


def heat_simulation(rho, init_proj, filter, proj):
    """
    Heat simulation given an input density. Material/void is seen as heat_eval sources. The heat_eval sinks are the
    points where the material/void should connect to.
    :param rho: Input density of the problem (design variable) [0, 1].
    :param resize_factor: Resizes the density in case the FEM-simulation is too big for the memory.
    :return: (Heat of the material, Heat of the void.
    """


    rho_p = proj(filter(init_proj(rho)))
    heat_sinks_matter = jnp.zeros((rho_p.shape[0] + 1,
                                   rho_p.shape[1] + 1,
                                   rho_p.shape[2] + 1), dtype='?')
    heat_sinks_matter = heat_sinks_matter.at[..., 0].set(True)
    kappa_r_matter = f2param(rho_p, ConfigSim.kappa)
    fem_matter = FEA3D_T(heat_sinks_matter)
    src_matter = jnp.pad(rho_p, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_matter = fem_matter.temperature(kappa_r_matter, src_matter)

    heat_sinks_void = jnp.zeros_like(heat_sinks_matter)
    heat_sinks_void = heat_sinks_void.at[0].set(True)
    heat_sinks_void = heat_sinks_void.at[-1].set(True)
    heat_sinks_void = heat_sinks_void.at[:, 0].set(True)
    heat_sinks_void = heat_sinks_void.at[:, -1].set(True)
    heat_sinks_void = heat_sinks_void.at[..., -1].set(True)
    kappa_r_void = f2param(1 - rho_p, ConfigSim.kappa)
    fem_void = FEA3D_T(heat_sinks_void)
    src_void = jnp.pad(1 - rho_p, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_void = fem_void.temperature(kappa_r_void, src_void)

    return jnp.sum(T_matter) / T_matter.size, jnp.sum(T_void) / T_void.size, kappa_r_matter


if __name__ == "__main__":
    import numpy as np
    resolution = 14
    rho_0 =  np.ones((int(np.ceil((ConfigSim.rho_shape[0] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil((ConfigSim.rho_shape[1] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil(ConfigSim.rho_shape[2] * resolution)))) * 1
    em_simulation(rho_0, 0, 14)
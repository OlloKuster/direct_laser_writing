import jax.numpy as jnp
import tidy3d as td
from tidy3d.plugins.autograd import rescale

from problems.power_splitter.simulation.config_structure import ConfigSim
from problems.power_splitter.simulation.sources_and_monitors import Sources, Monitors
from tofea.fea3d import FEA3D_T
from utility.helper import f2param


def make_sim_tidy(rho):
    input_waveguide = td.Structure(
        geometry=td.Box(center=(-(ConfigSim.lx - ConfigSim.wg_length) / 2 - 1,
                                0,
                                0),
                        size=(ConfigSim.wg_length + 4, ConfigSim.wg_width, ConfigSim.wg_width)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[1] ** 2)
    )
    output_waveguide = td.Structure(
        geometry=td.Box(center=((ConfigSim.lx - ConfigSim.wg_length) / 2 + 1,
                                ConfigSim.rho_size[1] / 4,
                                ConfigSim.rho_size[2] / 4),
                        size=(ConfigSim.wg_length + 4, ConfigSim.wg_width, ConfigSim.wg_height)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[1] ** 2)
    )

    eps = rescale(rho[:, rho.shape[1] // 2:, rho.shape[2] // 2:], ConfigSim.refr_index[0] ** 2,
                  ConfigSim.refr_index[1] ** 2)

    custom_structure = td.Structure.from_permittivity_array(
        geometry=td.Box(
            center=(0,
                    ConfigSim.rho_size[1] / 4,
                    ConfigSim.rho_size[2] / 4),
            size=(ConfigSim.rho_size[0], ConfigSim.rho_size[1] / 2, ConfigSim.rho_size[2] / 2)),
        eps_data=eps.reshape(eps.shape[0], eps.shape[1], eps.shape[2]))

    design_region_mesh = td.MeshOverrideStructure(
        geometry=custom_structure.geometry,
        dl=[1 / ConfigSim.dl] * 3,
        enforce=True,
    )

    grid_spec = td.GridSpec.auto(
        wavelength=ConfigSim.wavelength,
        min_steps_per_wvl=ConfigSim.min_steps_per_wvl
    )

    sim = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=[custom_structure, input_waveguide, output_waveguide],
        sources=[Sources.source],
        monitors=[Monitors.mode_monitor, Monitors.flux_monitor, Monitors.field_monitor_source, Monitors.field_monitor_center,
                  Monitors.eps_monitor],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.refr_index[0] ** 2),
        symmetry=(0, -1, 1)
    )

    grid_spec = sim.grid_spec.updated_copy(
        override_structures=list(sim.grid_spec.override_structures)
                            + [design_region_mesh]
    )

    return sim.updated_copy(grid_spec=grid_spec)


def heat_simulation(rho):
    """
    Heat simulation given an input density. Material/void is seen as heat_eval sources. The heat_eval sinks are the
    points where the material/void should connect to.
    :param rho: Input density of the problem (design variable) [0, 1].
    :param resize_factor: Resizes the density in case the FEM-simulation is too big for the memory.
    :return: (Heat of the material, Heat of the void.
    """
    rho_n = rho[:, rho.shape[1] // 2:, rho.shape[2] // 2:]

    heat_sinks_matter = jnp.zeros((rho_n.shape[0] + 1,
                                   rho_n.shape[1] + 1,
                                   rho_n.shape[2] + 1), dtype='?')

    resolution = rho.shape[0] // ConfigSim.rho_size[0]

    wg_width = int(jnp.ceil(ConfigSim.wg_width * resolution))

    heat_sinks_matter = heat_sinks_matter.at[resolution, :wg_width, :wg_width].set(True)
    kappa_r_matter = f2param(rho_n, ConfigSim.kappa)
    fem_matter = FEA3D_T(heat_sinks_matter)
    src_matter = jnp.pad(rho_n, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_matter = fem_matter.temperature(kappa_r_matter, src_matter)

    heat_sinks_void = jnp.zeros_like(heat_sinks_matter)
    heat_sinks_void = heat_sinks_void.at[0].set(True)
    heat_sinks_void = heat_sinks_void.at[-1].set(True)
    heat_sinks_void = heat_sinks_void.at[:, 0].set(True)
    heat_sinks_void = heat_sinks_void.at[:, -1].set(True)
    heat_sinks_void = heat_sinks_void.at[..., 0].set(True)
    heat_sinks_void = heat_sinks_void.at[..., -1].set(True)
    kappa_r_void = f2param(1 - rho_n, ConfigSim.kappa)
    fem_void = FEA3D_T(heat_sinks_void)
    src_void = jnp.pad(1 - rho_n, [(0, 1), (0, 1), (0, 1)], mode='constant', constant_values=0)
    T_void = fem_void.temperature(kappa_r_void, src_void)

    return jnp.sum(T_matter) / T_matter.size, jnp.sum(T_void) / T_void.size, kappa_r_matter

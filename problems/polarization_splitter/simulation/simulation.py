import jax
import jax.numpy as jnp
import autograd.numpy as anp
import tidy3d as td
from tidy3d.plugins.autograd import make_filter_and_project, rescale

from problems.polarization_splitter.simulation.config_structure import ConfigSim
from problems.polarization_splitter.simulation.sources_and_monitors import Sources, Monitors
from tofea.fea3d import FEA3D_T
from utility.helper import f2param, split_int


def make_sim_tidy(rho):
    input_waveguide = td.Structure(
        geometry=td.Box(center=(-(ConfigSim.lx - ConfigSim.wg_length) / 2 - 1,
                                -(ConfigSim.rho_size[1] - ConfigSim.wg_init_width) / 2 + 2*ConfigSim.buffer_side,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_init_height / 2),
                        size=(ConfigSim.wg_length + 4, ConfigSim.wg_init_width, ConfigSim.wg_init_height)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
    )
    output_waveguide_te = td.Structure(
        geometry=td.Box(center=((ConfigSim.lx - ConfigSim.wg_length) / 2 + 1,
                                -(ConfigSim.rho_size[1] - ConfigSim.wg_width) / 2 + 2*ConfigSim.buffer_side,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
                        size=(ConfigSim.wg_length + 4, ConfigSim.wg_width, ConfigSim.wg_height)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
    )

    output_waveguide_tm = td.Structure(
        geometry=td.Box(center=((ConfigSim.lx - ConfigSim.wg_length) / 2 + 1,
                                (ConfigSim.rho_size[1] - ConfigSim.wg_width) / 2  - 2*ConfigSim.buffer_side,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_width / 2),
                        size=(ConfigSim.wg_length + 4, ConfigSim.wg_height, ConfigSim.wg_width)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
    )

    substrate = td.Structure(
        geometry=td.Box(center=(0, 0, (-ConfigSim.lz + ConfigSim.thickness_substrate) / 2 - 1),
                        size=(td.inf, td.inf, ConfigSim.thickness_substrate + 2)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[1] ** 2)
    )
    eps = rescale(rho, ConfigSim.refr_index[0] ** 2, ConfigSim.refr_index[2] ** 2)

    custom_structure = td.Structure.from_permittivity_array(
        geometry=td.Box(
            center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.rho_size[2] / 2),
            size=(ConfigSim.rho_size[0], ConfigSim.rho_size[1], ConfigSim.rho_size[2])),
        eps_data=eps.reshape(eps.shape[0], eps.shape[1], eps.shape[2]))

    design_region_mesh = td.MeshOverrideStructure(
        geometry=custom_structure.geometry,
        dl=[1 / ConfigSim.dl] * 3,
        enforce=True,
    )

    grid_spec = td.GridSpec.uniform(
        dl=ConfigSim.dl
    )

    sim_te = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=[custom_structure, input_waveguide, output_waveguide_te, output_waveguide_tm, substrate],
        sources=[Sources.source_te],
        monitors=[Monitors.mode_monitor_te, Monitors.mode_monitor_tm, Monitors.field_monitor_source,
                  Monitors.field_monitor_center, Monitors.eps_monitor],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.refr_index[0] ** 2),

    )

    sim_tm = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=[custom_structure, input_waveguide, output_waveguide_te, output_waveguide_tm, substrate],
        sources=[Sources.source_tm],
        monitors=[Monitors.mode_monitor_te, Monitors.mode_monitor_tm, Monitors.field_monitor_source,
                  Monitors.field_monitor_center, Monitors.eps_monitor],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.refr_index[0] ** 2),

    )
    grid_spec_te = sim_te.grid_spec.updated_copy(
        override_structures=list(sim_te.grid_spec.override_structures)
                            + [design_region_mesh]
    )
    grid_spec_tm = sim_tm.grid_spec.updated_copy(
        override_structures=list(sim_tm.grid_spec.override_structures)
                            + [design_region_mesh]
    )

    return (sim_te, sim_tm)


def heat_simulation(rho, resize_factor):
    """
    Heat simulation given an input density. Material/void is seen as heat_eval sources. The heat_eval sinks are the
    points where the material/void should connect to.
    :param rho: Input density of the problem (design variable) [0, 1].
    :param resize_factor: Resizes the density in case the FEM-simulation is too big for the memory.
    :return: (Heat of the material, Heat of the void.
    """
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

    return jnp.sum(T_matter) / T_matter.size, jnp.sum(T_void) / T_void.size, kappa_r_matter

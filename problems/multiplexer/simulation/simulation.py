import jax
import jax.numpy as jnp
import autograd.numpy as anp
import tidy3d as td
from tidy3d import C_0
from tidy3d.plugins.autograd import make_filter_and_project, rescale

from problems.multiplexer.simulation.config_structure import ConfigSim
from problems.multiplexer.simulation.sources_and_monitors import Sources, Monitors
# from tofea.fea3d import FEA3D_T
from utility.helper import f2param


def make_sim_tidy(rho):
    input_waveguide = td.Structure(
        geometry=td.Box(center=(-(ConfigSim.lx - ConfigSim.wg_length) / 2,
                                0,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
                        size=(ConfigSim.wg_length + 5, ConfigSim.wg_width, ConfigSim.wg_height)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
    )

    output_waveguides = []
    output_monitors = []
    for i in range(ConfigSim.number_wgs):
        output_waveguide_1 = td.Structure(
            geometry=td.Box(center=((ConfigSim.lx - ConfigSim.wg_length) / 2,
                                    (i + 1) * ConfigSim.wg_dist,
                                    -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
                            size=(ConfigSim.wg_length + 5, ConfigSim.wg_width, ConfigSim.wg_height)),
            medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
        )

        output_waveguide_2 = td.Structure(
            geometry=td.Box(center=((ConfigSim.lx - ConfigSim.wg_length) / 2,
                                    - (i + 1) * ConfigSim.wg_dist / 2 - i * ConfigSim.wg_width,
                                    -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
                            size=(ConfigSim.wg_length + 5, ConfigSim.wg_width, ConfigSim.wg_height)),
            medium=td.Medium(permittivity=ConfigSim.refr_index[2] ** 2)
        )

        output_waveguides.append(output_waveguide_1)
        output_waveguides.append(output_waveguide_2)

        output_monitor_1 = td.ModeMonitor(
            center=((ConfigSim.lx - ConfigSim.wg_length) / 2,
                    (i + 1) * ConfigSim.wg_dist,
                    -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
            size=ConfigSim.size_monitor,
            freqs=ConfigSim.eval_freqs,
            mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
            name=f'Mode Monitor {ConfigSim.eval_wvls[i]}'
        )

        output_monitor_2 = td.ModeMonitor(
            center=((ConfigSim.lx - ConfigSim.wg_length) / 2,
                    - (i + 1) * ConfigSim.wg_dist,
                    -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
            size=ConfigSim.size_monitor,
            freqs=ConfigSim.eval_freqs,
            mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
            name=f'Mode Monitor {ConfigSim.eval_wvls[-i - 1]}'
        )

        output_monitors.append(output_monitor_1)
        output_monitors.append(output_monitor_2)

    substrate = td.Structure(
        geometry=td.Box(center=(0, 0, (-ConfigSim.lz + ConfigSim.thickness_substrate) / 2 + 0.5),
                        size=(td.inf, td.inf, ConfigSim.thickness_substrate - 1)),
        medium=td.Medium(permittivity=ConfigSim.refr_index[1] ** 2)
    )
    filter_project = make_filter_and_project(2 * ConfigSim.min_feature_size,
                                             ConfigSim.rho_size[2] / ConfigSim.lz)
    rho_filt_proj = filter_project(rho, 1)
    eps = rescale(rho_filt_proj, ConfigSim.refr_index[0] ** 2, ConfigSim.refr_index[2] ** 2)

    custom_structure = td.Structure.from_permittivity_array(
        geometry=td.Box(
            center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.rho_size[2] / 2),
            size=(ConfigSim.rho_size[0], ConfigSim.rho_size[1], ConfigSim.rho_size[2])),
        eps_data=eps.reshape(eps.shape[0], eps.shape[1], eps.shape[2]))

    design_region_mesh = td.MeshOverrideStructure(
        geometry=td.Box(size=(ConfigSim.rho_size[0], ConfigSim.rho_size[1], ConfigSim.rho_size[2])),
        dl=[ConfigSim.nx, ConfigSim.ny, ConfigSim.nz],
        enforce=True,
    )

    grid_spec = td.GridSpec.uniform(dl=1 / ConfigSim.dl)
    sim = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=output_waveguides + [input_waveguide, substrate, custom_structure],
        sources=[Sources.source],
        monitors=[Monitors.field_monitor_center, Monitors.eps_monitor] + output_monitors,
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.refr_index[0] ** 2),

    )

    return sim


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

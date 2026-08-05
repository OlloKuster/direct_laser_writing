import h5py
import pyvista as pv
import numpy as np
from cmcrameri import cm
import tidy3d as td
from tidy3d import web
from tidy3d.plugins.autograd import rescale

from problems.power_splitter.simulation.config_structure import ConfigSim
from problems.power_splitter.simulation.sources_and_monitors import Sources, Monitors

with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/plots/init_structures/power_splitter.h5") as f:
        grp = f["power_splitter"]
        eps = grp["eps"][:]
with h5py.File("/scratch/local/okuster/data/dlw/power_splitter/plots/data_0_inf.h5") as f:
        grp = f["power_splitter"]
        rho = grp["rho"][:]
        eps_final = grp["eps_normal"][:]

sim_data = td.SimulationData.from_file("/scratch/local/okuster/data/dlw/power_splitter/plots/progression/current_simulation.hdf5")
Ex = sim_data["Field Monitor"].Ex.squeeze().values
Ey = sim_data["Field Monitor"].Ey.squeeze().values
Ez = sim_data["Field Monitor"].Ez.squeeze().values
E = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)


pv.global_theme.allow_empty_mesh = True
eps = eps[:, eps.shape[1]//2:, eps.shape[2]//2:]
eps = np.concatenate((np.flip(eps, axis=1), eps), axis=1)
eps = np.concatenate((np.flip(eps, axis=2), eps), axis=2)

diff_x = E.shape[0] - eps.shape[0]
diff_y = E.shape[1] - eps.shape[1]
diff_z = E.shape[2] - eps.shape[2]
E = E[diff_x//2:-diff_x//2, diff_y//2:-diff_y//2, diff_z//2:-diff_z//2]

print(E.shape)
print(eps.shape)
data_nothing = pv.wrap(eps)

p = pv.Plotter(off_screen=True)
p.add_mesh(data_nothing.contour(), cmap='binary')
# p.add_volume(data_e_nothing, cmap='RdBu')
p.camera_position = 'xy'
p.camera.elevation = 30
p.camera.azimuth = 0
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/power_splitter_init.png')
p.close()

eps_final = eps_final[:, eps_final.shape[1]//2:, eps_final.shape[2]//2:]
eps_final = np.concatenate((np.flip(eps_final, axis=1), eps_final), axis=1)
eps_final = np.concatenate((np.flip(eps_final, axis=2), eps_final), axis=2)

print(f"printed shape: {eps_final.shape}")

data_nothing = pv.wrap(eps_final)
data_e = pv.wrap(np.clip(E[:], 1., 35))

p = pv.Plotter(off_screen=True)
p.add_mesh(data_nothing.contour(), cmap='binary')
p.add_volume(data_e, cmap='RdBu')
p.camera_position = 'xy'
p.camera.elevation = 30
p.camera.azimuth = 0
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/power_splitter_final_field.png')
p.close()

eps_final = eps_final[:, eps_final.shape[1]//2:, eps_final.shape[2]//2:]
eps_final = np.concatenate((np.flip(eps_final, axis=1), eps_final), axis=1)
eps_final = np.concatenate((np.flip(eps_final, axis=2), eps_final), axis=2)


data_nothing = pv.wrap(eps_final)
p = pv.Plotter(off_screen=True)
p.add_mesh(data_nothing.contour(), cmap='binary')
p.camera_position = 'xy'
p.camera.elevation = 30
p.camera.azimuth = 0
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/power_splitter_final.png')
p.close()


import tidy3d as td

resolution = 14
mask = np.ones_like(rho)

mask[:int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
mask[-int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
mask[:, :int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
mask[:, -int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
mask[:, :, :int(np.ceil(ConfigSim.buffer_top * resolution))] = 0
mask[:, :, -int(np.ceil(ConfigSim.buffer_top * resolution)):] = 0

rho = np.round(rho*mask)

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
        monitors=[Monitors.mode_monitor, Monitors.flux_monitor, Monitors.field_monitor_source,
                  Monitors.field_monitor_center,
                  Monitors.eps_monitor_full],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.refr_index[0] ** 2),
        symmetry=(0, -1, 1)
)

grid_spec = sim.grid_spec.updated_copy(
    override_structures=list(sim.grid_spec.override_structures)
                        + [design_region_mesh]  # + [waveguide_mesh]
)

sim = sim.updated_copy(grid_spec=grid_spec)
sim_data = web.run(sim, verbose=True)

path = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/plots/init_structures/"
# sim_data = web.load("fdve-f355bea1-d82a-4b94-93dd-9bbafed7877e", path=path + "_simulation_results.hdf5", verbose=True)
eps = sim_data["Permittivity Monitor Full"].eps_xx.real.squeeze()


print(f"init shape: {eps.shape}")

data = pv.wrap(np.array(eps[14:-14, 14:-14, 14:-14]))
p = pv.Plotter(off_screen=True)
p.add_mesh(data.contour(), cmap='binary')
p.camera_position = 'xy'
p.camera.elevation = 30
p.camera.azimuth = 0
p.remove_scalar_bar()
p.camera.zoom(1.3)
p.show(screenshot='plots/init_structures/power_splitter_rho.png')
p.close()


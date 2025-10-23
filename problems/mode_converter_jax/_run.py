import numpy as np
import jax
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
import h5py
import meep as mp

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from mode_calculator.meep_eigenmode_calculator_3D import find_mode_profile
from optimizer.optimizer import optimiser
from problems.mode_converter_jax.simulation._objective_loader import objective_loader
from problems.mode_converter_jax.simulation.config_structure import ConfigSim
from problems.mode_converter_jax.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()

    objectives = ["em_only", "em_only", "em_only"]

    filters = ["conic_jax", "conic_jax", "conic_jax"]
    filter_values = [resolution / 4 / np.sqrt(3), resolution / 4 / np.sqrt(3), resolution / 4 / np.sqrt(3)]

    projections = ["ssp_jax", "ssp_jax", "ssp_jax"]
    projection_values = [0.5, 0.5, ConfigPrint.rho_th_GT]

    optimizers = ["jax", "jax", "jax"]

    conversions = ["None", "None", "None"]
    backconversions = ["None", "None", "None"]

    rho_0 = np.ones((ConfigSim.rho_shape[0] * resolution,
                     ConfigSim.rho_shape[1] * resolution,
                     ConfigSim.rho_shape[2] * resolution)) * 0.5

    currents = find_mode_profile(ConfigSim.simulation_domain_shape, resolution,
                                 (ConfigSim.wg_width, ConfigSim.wg_width),
                                 ConfigSim.epsilon, ConfigSim.wavelength,
                                 d_sub=ConfigSim.buffer_sub + ConfigSim.dpml,
                                 mode=1, parity=mp.EVEN_Y, field=mp.Ez)

    currents = np.reshape(currents, (1, currents.shape[0], currents.shape[1]))

    target_field = find_mode_profile(ConfigSim.simulation_domain_shape, resolution,
                                     (ConfigSim.wg_width, ConfigSim.wg_width),
                                     ConfigSim.epsilon, ConfigSim.wavelength,
                                     d_sub=ConfigSim.buffer_sub + ConfigSim.dpml,
                                     mode=1, parity=mp.EVEN_Z + mp.EVEN_Y, field=mp.Ez)
    target_field = np.reshape(target_field, (1, target_field.shape[0], target_field.shape[1]))

    pos_y = int(jnp.ceil((ConfigSim.dpml + ConfigSim.buffer_sub + ConfigSim.wg_width / 2) * resolution))

    pos_x = int(jnp.ceil((ConfigSim.monitor_pos[0] * resolution) * resolution))

    loss_hist = []

    for i in range(len(betas)):
        objective = objective_loader(objectives[i], currents, resolution, target_field)
        if filters[i] == "dose_conv" and filters[i - 1] == "gauss_jax":
            proj_temp = projection_loader("tanh_jax", projection_values[i - 1], betas[i], resolution)
            rho_0 = proj_temp(filter(rho_0))

        filter = filter_loader(filters[i], filter_values[i])
        projection = projection_loader(projections[i], projection_values[i], betas[i], resolution)

        rho_0, loss = optimiser(rho_0, objective, filter, projection, optimizers[i])

        loss_hist += loss

        rho_0 = convert_to(rho_0, conversions[i])
        rho_opt_filtered = filter(rho_0)
        rho_opt_filtered = convert_to(rho_opt_filtered, backconversions[i])
        rho_opt_proj = projection(jnp.array(rho_opt_filtered))
        E, eps = em_simulation(jnp.array(rho_opt_proj), currents, resolution)
        plt.plot(loss_hist)
        # plt.yscale('log')
        plt.savefig(
            f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/loss_{betas[i]}.png")
        plt.close()
        plt.imshow(eps[:, pos_y].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[0], 0, ConfigSim.simulation_domain_shape[2]))
        plt.imshow(np.abs(E[2][:, pos_y].T), origin='lower', cmap="magma", alpha=0.8,
                   extent=(0, ConfigSim.simulation_domain_shape[0], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/eps_and_e_{betas[i]}.png")
        plt.close()

        plt.imshow(eps[pos_x].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[0], 0, ConfigSim.simulation_domain_shape[2]))
        plt.imshow(np.abs(E[2][pos_x].T), origin='lower', cmap="magma", alpha=0.8,
                   extent=(0, ConfigSim.simulation_domain_shape[0], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/eps_and_e_wg{betas[i]}.png")
        plt.close()

        plt.imshow(eps[:, pos_y].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[0], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/eps_{betas[i]}.png")
        plt.close()

        with h5py.File(
                f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/mode_converter_jax/plots/data_{betas[i]}.h5",
                'w') as f:
            grp = f.create_group("mode_converter")
            grp.create_dataset("E", data=E)
            grp.create_dataset("eps", data=eps)
            grp.create_dataset("rho", data=convert_to(rho_0, backconversions[i]))
            grp.create_dataset("loss", data=loss_hist)
            f.close()

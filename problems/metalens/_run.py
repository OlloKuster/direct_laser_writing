import numpy as np
import jax
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
import h5py

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimiser
from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader


def run(resolution, betas):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()

    rho_0 = np.ones((ConfigSim.rho_shape[0] * resolution,
                     ConfigSim.rho_shape[1] * resolution,
                     ConfigSim.rho_shape[2] * resolution)) * 0.5

    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)

    currents = jnp.ones(size_currents, jnp.complex128)

    objective_em = objective_loader("em_only", currents, resolution, 1)
    init_val_em = objective_em(rho_0)
    objective_heat = objective_loader("heat_only", currents, resolution, 1)
    init_T_mat, init_T_void = objective_heat(rho_0)

    init_val_mat = init_T_mat / ConfigSim.TARGET_MATERIAL
    init_val_void = init_T_void / ConfigSim.TARGET_VOID

    loss_hist = []

    for beta in betas:
        objective = objective_loader("em_heat", currents, resolution, init_val_em, init_val_mat, init_val_void)
        filter = filter_loader("dose_conv", resolution)
        projection = projection_loader("ssp_jax", ConfigPrint.rho_th_GT, beta, resolution)

        rho_0, loss = optimiser(rho_0, objective, filter, projection, "torch_jax")

        loss_hist += loss

        rho_opt_filtered = filter(torch.tensor(rho_0,
                                               device='cuda'))  # todo: build the conversions into the filter/projection filter themselves
        rho_opt_proj = projection(jnp.array(rho_opt_filtered.detach().cpu().numpy()))
        E, eps = em_simulation(jnp.array(rho_opt_proj), currents, resolution)
        plt.plot(loss_hist)
        plt.yscale('log')
        plt.savefig(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/loss_{beta}.png")
        plt.close()
        plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.imshow(np.abs(E[0][E[0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/eps_and_e_{beta}.png")
        plt.close()
        plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(f"/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/eps_{beta}.png")
        plt.close()

        with h5py.File("/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/data.h5",
                       'w') as f:
            grp = f.create_group("lens_3d")
            grp.create_dataset("E", data=E)
            grp.create_dataset("eps", data=eps)
            grp.create_dataset("rho", data=rho_0)
            grp.create_dataset("loss", data=loss_hist)
            f.close()


run(14, [16, 32, jnp.inf])

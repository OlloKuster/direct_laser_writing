import numpy as np
import jax
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimiser, optimizer_optax
from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, load=None, eval=False):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()

    objectives = "robust_em_heat"

    filters = "dose_conv"
    filter_values = resolution

    projections = "robust_ssp_jax"
    projection_values = [0.9*ConfigPrint.rho_th_GT, ConfigPrint.rho_th_GT, 1.1*ConfigPrint.rho_th_GT]

    optimizers = "torch_jax"

    conversions = "torch"
    backconversions = "torch2np"

    rho_0 = np.ones((ConfigSim.rho_shape[0] * resolution,
                     ConfigSim.rho_shape[1] * resolution,
                     ConfigSim.rho_shape[2] * resolution)) * ConfigPrint.rho_th_GT

    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)

    currents = jnp.ones(size_currents, jnp.complex128)

    objective_em = objective_loader("em_only", currents, resolution, 1, 1, 1)
    init_val_em, _ = objective_em(np.ones_like(rho_0)*0.5)
    objective_heat = objective_loader("heat_only")
    init_T_mat, init_T_void = objective_heat(np.ones_like(rho_0)*0.5)

    init_val_mat = init_T_mat / ConfigSim.TARGET_MATERIAL
    init_val_void = init_T_void / ConfigSim.TARGET_VOID

    if load is not None:
        f = h5py.File(load)
        grp = f["lens_3d"]
        rho_0 = grp["rho"][:]
        f.close()

    loss_hist = []
    em_loss_hist = []

    for i in range(len(betas)):
        objective = objective_loader(objectives, currents, resolution, init_val_em, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values)
        projection = projection_loader(projections, projection_values, betas[i], resolution)

        rho_0, loss, em_loss, grads = optimiser(rho_0, objective, filter, projection, optimizers, eval=eval)

        loss_hist += loss
        em_loss_hist += em_loss

        rho_0 = convert_to(rho_0, conversions)
        rho_opt_filtered = filter(rho_0)
        rho_opt_filtered = convert_to(rho_opt_filtered, backconversions)
        rho_opt_proj = projection(jnp.array(rho_opt_filtered))
        E, eps = em_simulation(jnp.array(rho_opt_proj), currents, resolution)
        plt.plot(loss_hist)
        # plt.yscale('log')
        plt.savefig(
            f"problems/metalens/plots/loss_{betas[i]}.png")
        plt.close()
        plt.plot(em_loss_hist[0], legend='eroded')
        plt.plot(em_loss_hist[1], legend='normal')
        plt.plot(em_loss_hist[2], legend='dilated')
        plt.legend()
        plt.savefig(
            f"problems/metalens/plots/em_loss_{betas[i]}.png")
        plt.close()
        plt.plot(grads)
        plt.savefig(
            f"problems/metalens/plots/grads_{betas[i]}.png")
        plt.xlabel("Iteration")
        plt.ylabel("Gradient")
        plt.close()
        plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.imshow(np.abs(E[0][E[0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"problems/metalens/plots/eps_and_e_{betas[i]}.png")
        plt.close()
        plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                   extent=(0, ConfigSim.simulation_domain_shape[1], 0, ConfigSim.simulation_domain_shape[2]))
        plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(
            f"problems/metalens/plots/eps_{betas[i]}.png")
        plt.close()

        with h5py.File(
                f"problems/metalens/plots/data_{betas[i]}.h5",
                'w') as f:
            grp = f.create_group("lens_3d")
            grp.create_dataset("E", data=E)
            grp.create_dataset("eps", data=eps)
            grp.create_dataset("rho", data=convert_to(rho_0, backconversions))
            grp.create_dataset("loss", data=loss_hist)
            grp.create_dataset("em_loss", data=em_loss_hist)
            f.close()

        rho_0 = convert_to(rho_0, backconversions)

import numpy as np
import jax
import jax.numpy as jnp
import scipy
import torch
import h5py

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.mode_converter.simulation._objective_loader import objective_loader
from problems.mode_converter.simulation.config_structure import ConfigSimMode
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, setting: dict, loss_hist, em_loss_hist,opt, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()


    objectives = setting["objectives"]
    filters = setting["filters"]
    filter_values = setting["filter_factor"] * resolution
    projections = setting["projection"]
    init_projections = setting["init_projection"]
    projection_values = setting["projection_values"]
    init_projection_values = setting["init_projection_values"]
    plotter_eval_name = setting["plotter_eval"]
    plotter_final_name = setting["plotter_final"]
    optimizers = setting["optimizers"]

    plotter_eval = plot_loader(plotter_eval_name)
    plotter_final = plot_loader(plotter_final_name)




    rho_0 = np.ones((ConfigSimMode.nx, ConfigSimMode.ny, ConfigSimMode.nz)) * 0.5

    # mask = np.ones_like(rho_0)
    # mask[:int(ConfigSimMode.buffer_side * resolution)] = 0
    # mask[:, :int(ConfigSimMode.buffer_side * resolution)] = 0
    # mask[:, :, -int(ConfigSimMode.buffer_top * resolution):] = 0

    objective_heat = objective_loader("heat_only")
    init_T_mat, init_T_void = objective_heat(np.ones_like(rho_0) * 0.5)

    init_val_mat = init_T_mat / ConfigSimMode.TARGET_MATERIAL
    init_val_void = init_T_void / ConfigSimMode.TARGET_VOID

    for i in range(len(betas)):
        if not full_bin:
            beta_ssp = betas[i]
        else:
            beta_ssp = np.inf
        objective = objective_loader(objectives, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values)
        projection = projection_loader(projections, projection_values, beta_ssp, resolution)
        init_projection = projection_loader(init_projections, init_projection_values, betas[i], resolution)

        if opt == "optax":
            rho_0, loss, em_loss, grads = optimizer_optax(rho_0, objective, 1, filter, projection, init_projection,
                                                          plotter_eval, optimizers,
                                                          eval=eval)
        elif opt == "nlopt":
            rho_0, loss, em_loss, grads = optimizer_nlopt(rho_0, objective, 1, filter, projection, init_projection,
                                                          plotter_eval, optimizers,
                                                          eval=eval)
        else:
            print("no valid optimizer chosen")
            return

        loss_hist += loss
        em_loss_hist += em_loss

    return loss_hist, em_loss_hist
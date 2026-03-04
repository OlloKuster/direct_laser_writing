import numpy as np
import jax
import jax.numpy as jnp
import scipy
import torch
import h5py
import tidy3d as td

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.polarization_splitter.simulation._objective_loader import objective_loader
from problems.polarization_splitter.simulation.config_structure import ConfigSim
from problems.polarization_splitter.simulation.simulation import make_sim_tidy
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, setting: dict, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False,
        full_bin=False,
        run_id=0):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()

    np.random.seed(42)

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
    conversions = setting["conversions"]
    backconversions = setting["backconversions"]

    plotter_eval = plot_loader(plotter_eval_name)
    plotter_final = plot_loader(plotter_final_name)

    rho_0 = np.ones((ConfigSim.nx, ConfigSim.ny, ConfigSim.nz)) * 0.5
    # rho_0[:, :rho_0.shape[1]//2] = 0.3
    #
    rho_0 = np.random.rand(ConfigSim.nx, ConfigSim.ny,
                           ConfigSim.nz)
    #
    # rho_0 = np.repeat(rho_0, ConfigSim.nz, axis=2)

    # rho_0 = np.round(scipy.ndimage.gaussian_filter(np.random.rand(ConfigSim.nx,
    #                                                               ConfigSim.ny,
    #                                                               1), sigma=0.25 * ConfigSim.nx))
    #
    # rho_0 = np.repeat(rho_0, ConfigSim.nz, axis=2)

    start_wg = int(np.ceil((ConfigSim.rho_size[1] - ConfigSim.wg_width) / 2 * resolution))
    end_wg = int(np.ceil((ConfigSim.rho_size[1] + ConfigSim.wg_width) / 2 * resolution))

    mask = np.ones_like(rho_0)
    # mask[:int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl)), :start_wg,
    # :int(np.ceil(ConfigSim.wg_height * ConfigSim.dl))] = 0
    # mask[:int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl)), end_wg:,
    # :int(np.ceil(ConfigSim.wg_height * ConfigSim.dl))] = 0
    # mask[-int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl)):, :start_wg,
    # :int(np.ceil(ConfigSim.wg_height * ConfigSim.dl))] = 0
    # mask[-int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl)):, end_wg:,
    # :int(np.ceil(ConfigSim.wg_height * ConfigSim.dl))] = 0
    # mask[:, :int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl))] = 0
    # mask[:, -int(np.ceil(ConfigSim.buffer_side * ConfigSim.dl)):] = 0
    # mask[:, :, -int(np.ceil(ConfigSim.buffer_top * ConfigSim.dl)):] = 0
    mask[:int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
    mask[-int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
    mask[:, :int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
    mask[:, -int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
    mask[:, :, -int(np.ceil(ConfigSim.buffer_top * resolution)):] = 0

    objective_heat = objective_loader("heat_only")
    init_T_mat, init_T_void = objective_heat(np.ones_like(rho_0) * 0.5)

    init_val_mat = init_T_mat / ConfigSim.TARGET_MATERIAL
    init_val_void = init_T_void / ConfigSim.TARGET_VOID

    if load is not None:
        f = h5py.File(load)
        grp = f["polarization_splitter"]
        rho_0 = grp["rho"][:]
        f.close()

    for i in range(len(betas)):
        print(f"beta: {betas[i]}")
        objective = objective_loader(objectives, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values)
        projection = projection_loader(projections, projection_values, betas[i], resolution)
        init_projection = projection_loader(init_projections, init_projection_values, betas[i], resolution)

        if opt == "optax":
            rho_0, loss, em_loss, grads = optimizer_optax(rho_0, objective, mask, filter, projection, init_projection,
                                                          plotter_eval, optimizers, max_evals=max_evals,
                                                          eval=eval)
        elif opt == "nlopt":
            rho_0, loss, em_loss, grads = optimizer_nlopt(rho_0, objective, mask, filter, projection, init_projection,
                                                          plotter_eval, optimizers, max_evals=max_evals,
                                                          eval=eval)
        else:
            print("no valid optimizer chosen")
            return

        loss_hist += loss
        em_loss_hist += em_loss

        rho_proj_init = init_projection(rho_0) * mask
        rho_0 = convert_to(rho_proj_init, conversions)
        rho_opt_filtered = filter(rho_0)
        rho_opt_filtered = convert_to(rho_opt_filtered, backconversions)
        rho_opt_proj = projection(jnp.array(rho_opt_filtered))

        sim = make_sim_tidy(np.array(rho_opt_proj))
        eps = np.abs(sim[0].epsilon(td.Box(
            center=(0, 0, 0),
            size=(ConfigSim.lx, ConfigSim.ly, ConfigSim.lz)
        )))

        plotter_final(extent=(ConfigSim.ly, ConfigSim.lz),
                      rho_0=convert_to(rho_0, backconversions),
                      loss_hist=loss_hist,
                      beta=betas[i],
                      em_loss_hist=em_loss_hist,
                      eps=eps,
                      run_id=run_id,
                      save=True)

        rho_0 = convert_to(rho_0, backconversions)

    return loss_hist, em_loss_hist

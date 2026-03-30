import numpy as np
import jax
import jax.numpy as jnp
import torch
import h5py
import tidy3d as td

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.power_splitter.simulation._objective_loader import objective_loader
from problems.power_splitter.simulation.config_structure import ConfigSim
from problems.power_splitter.simulation.simulation import make_sim_tidy
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, setting: dict, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False,
        full_bin=False,
        run_id=0):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()

    np.random.seed(42)

    init_value = 0.5

    objectives = setting["objectives"]
    filters = setting["filters"]
    filter_values = setting["filter_factor"] * resolution
    lp_deviation = setting["lp_deviation"]
    if lp_deviation == 0:
        lp_deviation = ConfigPrint.lp
    projections = setting["projection"]
    init_projections = setting["init_projection"]
    projection_values = setting["projection_values"]
    init_projection_values = setting["init_projection_values"]
    plotter_eval_name = setting["plotter_eval"]
    plotter_final_name = setting["plotter_final"]
    optimizers = setting["optimizers"]
    conversions = setting["conversions"]
    backconversions = setting["backconversions"]

    target_material = setting["target_material"]
    target_void = setting["target_void"]

    plotter_eval = plot_loader(plotter_eval_name)
    plotter_final = plot_loader(plotter_final_name)

    rho_0 = np.ones((ConfigSim.nx, ConfigSim.ny, ConfigSim.nz)) * init_value
    # rho_0[:, ConfigSim.ny//2-resolution//3:ConfigSim.ny//2+resolution//3] = 0
    # rho_0[:, rho_0.shape[1]//2 - resolution:rho_0.shape[1]//2+resolution] = 0
    # rho_0[:, :rho_0.shape[1]//2] = 0.3
    #
    # rho_0 = np.random.rand(ConfigSim.nx, ConfigSim.ny,
    #                        ConfigSim.nz)
    #
    # rho_0 = np.repeat(rho_0, ConfigSim.nz, axis=2)

    # rho_0 = np.round(scipy.ndimage.gaussian_filter(np.random.rand(ConfigSim.nx,
    #                                                               ConfigSim.ny,
    #                                                               1), sigma=0.25 * ConfigSim.nx))
    #
    # rho_0 = np.repeat(rho_0, ConfigSim.nz, axis=2)

    mask = np.ones_like(rho_0)

    mask[:int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
    mask[-int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
    mask[:, :int(np.ceil(ConfigSim.buffer_side * resolution))] = 0
    mask[:, -int(np.ceil(ConfigSim.buffer_side * resolution)):] = 0
    mask[:, :, :int(np.ceil(ConfigSim.buffer_top * resolution))] = 0
    mask[:, :, -int(np.ceil(ConfigSim.buffer_top * resolution)):] = 0

    filter_0 = filter_loader(filters, filter_values, lp_deviation)
    init_projection_0 = projection_loader(init_projections, init_projection_values, betas[0], resolution)
    projection_0 = projection_loader(projections, projection_values, betas[0], resolution)

    objective_heat = objective_loader(setting["init_heat"])

    rho_0_init_bin = np.array(init_projection_0(np.ones_like(rho_0) * init_value)) * mask
    if optimizers == 'torch_jax':
        rho_0_init = filter_0((torch.tensor(rho_0_init_bin, device='cuda', requires_grad=True))).detach().cpu().numpy()
    if optimizers == "jax":
        rho_0_init = filter_0(rho_0_init_bin)
    rho_0_bin = projection_0(rho_0_init)
    if len(rho_0_bin) == 3:
        init_T_mat, init_T_void = objective_heat(rho_0_bin[1])
    else:
        init_T_mat, init_T_void = objective_heat(rho_0_bin)

    init_val_mat = (init_T_mat + 1e-3) / (1 + target_material)
    init_val_void = (init_T_void + 1e-3) / (1 + target_void)

    if load is not None:
        f = h5py.File(load)
        grp = f["power_splitter"]
        rho_0 = grp["rho"][:]
        f.close()

    for i in range(len(betas)):
        print(f"beta: {betas[i]}")
        objective = objective_loader(objectives, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values, lp_deviation)
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
        if type(rho_opt_proj) is tuple:

            sim_erosion = make_sim_tidy(np.array(rho_opt_proj[0]))
            sim_normal = make_sim_tidy(np.array(rho_opt_proj[1]))
            sim_dilation = make_sim_tidy(np.array(rho_opt_proj[2]))

            eps_erosion = np.abs(sim_erosion.epsilon(td.Box(
                center=(0, 0, 0),
                size=(ConfigSim.lx, ConfigSim.ly, ConfigSim.lz)
            )))
            eps_normal = np.abs(sim_normal.epsilon(td.Box(
                center=(0, 0, 0),
                size=(ConfigSim.lx, ConfigSim.ly, ConfigSim.lz)
            )))
            eps_dilation = np.abs(sim_dilation.epsilon(td.Box(
                center=(0, 0, 0),
                size=(ConfigSim.lx, ConfigSim.ly, ConfigSim.lz)
            )))

            eps = (eps_erosion, eps_normal, eps_dilation)
        else:
            sim = make_sim_tidy(np.array(rho_opt_proj))
            eps = np.abs(sim.epsilon(td.Box(
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

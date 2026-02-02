import numpy as np
import jax
import jax.numpy as jnp
import scipy
import torch
import h5py

from filtering._filter_loader import filter_loader
from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader
from utility.helper import convert_to
from optimizer import config as ConfigOpt


def run(resolution, betas, setting: dict, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False,
        full_bin=False,
        run_id=0):
    """
    Runs the optimization process. Lower level "main".
    :param resolution: Resolution of the problem.
    :param betas: List of the binarization levels.
    :param setting: Selects which problem should be optimized.
    :param load: Load an external rho.
    :param eval: Activate intermediate evaluation/plotting of rho.
    :return: None.
    """

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
    conversions = setting["conversions"]
    backconversions = setting["backconversions"]

    target_material = setting["target_material"]
    target_void = setting["target_void"]

    plotter_eval = plot_loader(plotter_eval_name)
    plotter_final = plot_loader(plotter_final_name)

    rho_0 = np.ones((int(np.ceil(ConfigSim.rho_shape[0] * resolution)),
                     int(np.ceil(ConfigSim.rho_shape[1] * resolution)),
                     int(np.ceil(ConfigSim.rho_shape[2] * resolution)))) * 0.5
    # rho_0[:, :, rho_0.shape[2]//2:] = 0
    #
    # rho_0 = np.round(scipy.ndimage.gaussian_filter(np.random.rand(ConfigSim.rho_shape[0] * resolution,
    #                        ConfigSim.rho_shape[1] * resolution,
    #                        ConfigSim.rho_shape[2] * resolution), sigma=0.3*resolution))

    mask = np.ones_like(rho_0)
    mask[:int(ConfigSim.buffer_side * resolution)] = 0
    mask[:, :int(ConfigSim.buffer_side * resolution)] = 0
    mask[:, :, -int(ConfigSim.buffer_top * resolution):] = 0

    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)

    currents = jnp.ones(size_currents, jnp.complex128)

    objective_em = objective_loader(setting["init_em"], currents, resolution, 1, 1, 1)
    init_val_em, _ = objective_em(jnp.ones_like(rho_0) * 0.5)
    objective_heat = objective_loader(setting["init_heat"])
    init_T_mat, init_T_void = objective_heat(np.ones_like(rho_0) * 0.5)

    init_val_mat = init_T_mat / target_material
    init_val_void = init_T_void / target_void

    if load is not None:
        f = h5py.File(load)
        grp = f["lens_3d"]
        rho_0 = grp["rho"][:]
        f.close()

    uptick = 0

    for i in range(len(betas)):
        if not full_bin:
            beta_ssp = np.round(betas[i], 2)
        else:
            beta_ssp = np.inf
        print(f"beta: {beta_ssp}")
        objective = objective_loader(objectives, currents, resolution, init_val_em, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values)
        projection = projection_loader(projections, projection_values, beta_ssp, resolution)
        init_projection = projection_loader(init_projections, init_projection_values, betas[i], resolution)

        if opt == "optax":
            rho_0, loss, em_loss, grads = optimizer_optax(rho_0, objective, mask, filter, projection, init_projection,
                                                          plotter_eval, optimizers, max_evals,
                                                          eval=eval)
        elif opt == "nlopt":
            rho_0, loss, em_loss, grads = optimizer_nlopt(rho_0, objective, mask, filter, projection, init_projection,
                                                          plotter_eval, optimizers, max_evals,
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
            E_erosion, eps_erosion = em_simulation(jnp.array(rho_opt_proj[0]), currents, resolution)
            E_normal, eps_normal = em_simulation(jnp.array(rho_opt_proj[1]), currents, resolution)
            E_dilation, eps_dilation = em_simulation(jnp.array(rho_opt_proj[2]), currents, resolution)
            E = (E_erosion, E_normal, E_dilation)
            eps = (eps_erosion, eps_normal, eps_dilation)
        else:
            E, eps = em_simulation(jnp.array(rho_opt_proj), currents, resolution)

        if beta_ssp == betas[-1]:
            save = True
        else:
            save = False
        plotter_final(extent=(ConfigSim.simulation_domain_shape[1], ConfigSim.simulation_domain_shape[2]),
                      rho_0=convert_to(rho_0, backconversions),
                      loss_hist=loss_hist,
                      beta=beta_ssp,
                      em_loss_hist=em_loss_hist,
                      grads=grads,
                      eps=eps,
                      E=E,
                      run_id=run_id,
                      save=save)

        rho_0 = convert_to(rho_0, backconversions)
    return loss_hist, em_loss_hist

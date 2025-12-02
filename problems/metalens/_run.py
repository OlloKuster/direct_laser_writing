import numpy as np
import jax
import jax.numpy as jnp
import scipy
import torch
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, setting: dict, load=None, eval=False):
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

    plotter_eval = plot_loader(plotter_eval_name)
    plotter_final = plot_loader(plotter_final_name)

    rho_0 = np.ones((ConfigSim.rho_shape[0] * resolution,
                     ConfigSim.rho_shape[1] * resolution,
                     ConfigSim.rho_shape[2] * resolution)) * 0.5

    # rho_0_0 = np.random.rand(ConfigSim.rho_shape[0] * resolution,
    #                  ConfigSim.rho_shape[1] * resolution,
    #                  ConfigSim.rho_shape[2] * resolution)
    # rho_0 = np.round(scipy.ndimage.gaussian_filter(rho_0_0, sigma=0.5*resolution))
    # plt.imshow(rho_0[rho_0.shape[0]//2])
    # plt.show()

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
        projection = projection_loader(projections, projection_values, np.inf, resolution)
        init_projection = projection_loader(init_projections, init_projection_values, betas[i], resolution)
        rho_0, loss, em_loss, grads = optimizer_nlopt(rho_0, objective, mask, filter, projection, init_projection,
                                                      plotter_eval, optimizers,
                                                      eval=eval)

        loss_hist += loss
        em_loss_hist += em_loss

        rho_0 = convert_to(rho_0, conversions)
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

        plotter_final(extent=(ConfigSim.simulation_domain_shape[1], ConfigSim.simulation_domain_shape[2]),
                      rho_0=rho_0.detach().cpu().numpy(),
                      loss_hist=loss_hist,
                      beta=betas[i],
                      em_loss_hist=em_loss_hist,
                      grads=grads,
                      eps=eps,
                      E=E,
                      save=True)

        rho_0 = convert_to(rho_0, backconversions)

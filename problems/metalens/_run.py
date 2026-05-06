import numpy as np
import jax
import jax.numpy as jnp
import torch
import h5py

from filtering._filter_loader import filter_loader
from optimizer.optimizer import optimizer_nlopt, optimizer_optax
from plotter._plot_loader import plot_loader
from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from problems.metalens.simulation.simulation import em_simulation
from projection._projection_loader import projection_loader
from utility.helper import convert_to


def run(resolution, betas, setting: dict, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, run_id=0):
    """
    Runs the optimization process. Lower level "main".
    :param resolution: Resolution of the problem.
    :param betas: List of the binarization levels.
    :param setting: Selects which problem should be optimized.
    :param load: Load an external rho.
    :param eval: Activate intermediate evaluation/plotting of rho.
    :return: None.
    """

    jax.config.update("jax_enable_x64", True)  # Might be redundant
    torch.cuda.empty_cache()

    # Setting up the optimization parameters
    objectives = setting["objectives"]
    filters = setting["filters"]
    filter_values = setting["filter_factor"] * resolution
    lp_deviation = setting["lp_deviation"]
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

    # Initial density setup
    init_value = 0.5

    rho_0 = np.ones((int(np.ceil((ConfigSim.rho_shape[0] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil((ConfigSim.rho_shape[1] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil(ConfigSim.rho_shape[2] * resolution)))) * init_value

    # Mask for forcing sides to be 0
    mask = np.ones_like(rho_0)
    mask[:int(ConfigSim.buffer_side * resolution)] = 0
    mask[:, :int(ConfigSim.buffer_side * resolution)] = 0
    mask[:, :, -int(ConfigSim.buffer_top * resolution):] = 0

    # Input current terms (Plane Wave)
    size_currents = (ConfigSim.currents_shape[0] * resolution,
                     ConfigSim.currents_shape[1] * resolution,
                     1)

    currents = jnp.ones(size_currents, jnp.complex128)

    # Calculating initial L_heat/void
    filter_0 = filter_loader(filters, filter_values, lp_deviation)
    projection_0 = projection_loader(projections, projection_values, betas[0], resolution)

    objective_em = objective_loader(setting["init_em"], currents, resolution, 1, 1, 1)
    init_val_em, _ = objective_em(jnp.ones_like(rho_0))
    objective_heat = objective_loader(setting["init_heat"])

    rho_0_init_bin = np.array(np.ones_like(rho_0)*init_value) * mask
    if optimizers == 'torch_jax':
        rho_0_init = filter_0((torch.tensor(rho_0_init_bin, device='cuda', requires_grad=True))).detach().cpu().numpy()
    if optimizers == "jax":
        rho_0_init = filter_0(rho_0_init_bin)
    rho_0_bin = projection_0(rho_0_init)
    if len(rho_0_bin) == 3:
        init_T_mat, init_T_void = objective_heat(rho_0_bin[1], resolution)
    else:
        init_T_mat, init_T_void = objective_heat(rho_0_bin, resolution)


    init_val_mat = (init_T_mat + 1e-3) / (1 + target_material)  # Small offset to avoid singularities
    init_val_void = (init_T_void + 1e-3) / (1 + target_void)

    if load is not None:
        f = h5py.File(load)
        grp = f["lens_3d"]
        rho_0 = grp["rho"][:]
        f.close()

    # Main Optimization loop
    for i in range(len(betas)):
        print(f"beta: {betas[i]}")

        objective = objective_loader(objectives, currents, resolution, init_val_em, init_val_mat, init_val_void)

        filter = filter_loader(filters, filter_values, lp_deviation)
        projection = projection_loader(projections, projection_values, betas[i], resolution)
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

        # Final manipulations to save the relevant details.
        rho_precomp = (
        init_projection((rho_0) * mask)[:-ConfigSim.buffer_side * resolution, :-ConfigSim.buffer_side * resolution])
        rho_proj_init = init_projection(rho_0) * mask
        rho_proj_init = convert_to(rho_proj_init, conversions)
        rho_opt_filtered = filter(rho_proj_init)
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

        if True:
            plotter_final(extent=(ConfigSim.simulation_domain_shape[1], ConfigSim.simulation_domain_shape[2]),
                          rho_0=rho_0,
                          rho_precomp=rho_precomp,
                          loss_hist=loss_hist,
                          beta=betas[i],
                          em_loss_hist=em_loss_hist,
                          grads=grads,
                          eps=eps,
                          E=E,
                          run_id=run_id,
                          save=True)

    return loss_hist, em_loss_hist

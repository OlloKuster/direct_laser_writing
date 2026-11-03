import jax
import numpy as np
import nlopt
import time

from optimizer import config


def optimizer_nlopt(rho, objective, mask, filter, projection, init_projection, plotter, mode, max_evals, eval=False):
    """
    NLOpt based optimizer for non-linear, gradient based optimization.
    The density is first put through init_projection to give a "precomensated" density. Then it is put
    through filter and projection to evaluate the performance (objective) of the "printed" structure.
    :param rho: Input density of the TopOpt problem (to be optmized).
    :param objective: Objective function of the problem.
    :param mask: Mask for setting regions to 0. This is done so that there is no hard cutoff at the edge of the design
                 region.
    :param filter: Filter function used in this optimization run.
    :param projection: Projection function used in this optimization run.
    :param init_projection: Initial projection function used in this optimization run.
    :param plotter: Plotting functions used.
    :param mode: Mode of the gradient evaulation, e.g. if a torch wrapper is required.
    :param eval: If intermediate plots will be plotted.
    :return: (rho_opt, loss_hist, em_hist, grad_hist) Tuple of the optimized density, history of the loss and the
             pure electromagnetic loss and history of the gradients.
    """

    loss_hist = []
    em_hist = []
    grad_hist = []
    cur_eps = []


    def f_jax(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        x = np.array(x * mask)
        value_em_sim, grad = jax.value_and_grad(objective, has_aux=True)(x)
        value = float(value_em_sim[0])  # Requires np float and not jax.numpy float
        em_hist.append(value_em_sim[1][0])
        cur_eps.append(value_em_sim[1][1])
        print(f"value: {value}")
        print(f"grad: {np.mean(grad)}")
        print(f"iteration: {config.cur_it + 1}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            plotter(x, _, cur_eps[-1], projection, config.cur_it)
        return value

    f = f_jax

    opt = nlopt.opt(config.OPTIMISER, rho.size)
    # opt.set_param('tolg', 1e-12)
    opt.set_max_objective(f)
    opt.set_maxeval(max_evals)
    # opt.set_ftol_abs(config.FTOL_ABS)
    # opt.set_ftol_rel(config.FTOL_REL)
    # opt.set_stopval(1e-4)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(rho.ravel())
    rho_opt = rho_opt.reshape(rho.shape)

    return rho_opt, loss_hist, em_hist, grad_hist

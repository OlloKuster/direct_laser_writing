import jax
import numpy as np
import torch
import nlopt
import optax
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

    def select_f(mode):
        if mode == "torch_jax":
            return f_torch_jax
        if mode == "jax":
            return f_jax

    class FomEmTorchF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            grad_em_sim_f = jax.value_and_grad(objective, has_aux=True)
            value_em_sim, grad_em_sim = grad_em_sim_f(projection(x.detach().cpu().numpy().astype(np.float64)))
            em_hist.append(value_em_sim[1][0])
            cur_eps.append(value_em_sim[1][1])
            # cur_field.append(value_em_sim[1][2])
            value_em_sim = value_em_sim[0]
            ctx.save_for_backward(torch.tensor(np.array(grad_em_sim), device='cuda', requires_grad=True))
            return torch.tensor(np.array(value_em_sim), device='cuda', requires_grad=True)

        @staticmethod
        def backward(ctx, grad):
            grad_em_sim, = ctx.saved_tensors
            return grad_em_sim

    def f_torch_jax(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        x = np.array(init_projection(x) * mask)
        rho_0 = torch.tensor(x, device='cuda', requires_grad=True)
        rho_final = filter(rho_0)
        fom = FomEmTorchF.apply(rho_final)
        fom.backward(retain_graph=True)
        grad_torch = rho_0.grad
        value = fom.detach().cpu().numpy()
        grad = grad_torch.detach().cpu().numpy()
        value = float(value)  # Requires np float and not jax.numpy float
        grad_hist.append(np.mean(grad))
        print(f"value: {value}")
        print(f"grad: {np.mean(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            plotter(x, rho_final.detach().cpu().numpy(), cur_eps[-1], projection, config.cur_it)
        return value

    def f_jax(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        x = np.array(init_projection(x) * mask)
        rho_final = filter(x)
        value_obj, grad = jax.value_and_grad(objective, has_aux=True)(projection(rho_final))
        value = float(value_obj[0])  # Requires np float and not jax.numpy float
        value_em = float(value_obj[1])
        print(f"value: {value}")
        print(f"grad: {np.mean(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        em_hist.append(value_em)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            plotter(x, rho_final, cur_eps, projection, config.cur_it)
        return value

    f = select_f(mode)

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


def optimizer_optax(rho, objective, mask, filter, projection, init_projection, plotter, mode, max_evals, eval=False):
    """
    Optax based optimizer for more machine learning based optimization.
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
    rho = np.array(rho)

    optimizer = optax.adam(learning_rate=config.lr)
    opt_state = optimizer.init(rho)

    loss_hist = []
    em_hist = []
    grad_hist = []

    rho_opt = np.array(rho)

    best_val = -1e3
    prev_val = 100

    class FomEmTorchF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            grad_em_sim_f = jax.value_and_grad(objective, has_aux=True)
            value_em_sim, grad_em_sim = grad_em_sim_f(projection(x.detach().cpu().numpy().astype(np.float64)))
            em_hist.append(value_em_sim[1])
            value_em_sim = value_em_sim[0]
            ctx.save_for_backward(torch.tensor(np.array(grad_em_sim), device='cuda', requires_grad=True))
            return torch.tensor(np.array(value_em_sim), device='cuda', requires_grad=True)

        @staticmethod
        def backward(ctx, grad):
            grad_em_sim, = ctx.saved_tensors
            return grad_em_sim

    for i in range(max_evals):
        start = time.time()
        rho_init = np.array(init_projection(rho) * mask)
        if mode == "jax":
            rho_final = filter(rho_init)
            value_obj, grad = jax.value_and_grad(objective, has_aux=True)(projection(rho_final))
            value = float(value_obj[0])  # Requires np float and not jax.numpy float
            value_em = float(value_obj[1])
            grad = np.array(grad)
            loss_hist.append(value)
            em_hist.append(value_em)

        if mode == "torch_jax":
            rho_0 = torch.tensor(rho_init, device='cuda', requires_grad=True)
            rho_final = filter(rho_0)
            fom = FomEmTorchF.apply(rho_final)
            fom.backward(retain_graph=True)
            grad_torch = rho_0.grad
            rho_0 = rho_0.detach().cpu().numpy()
            value = fom.detach().cpu().numpy()
            grad = grad_torch.detach().cpu().numpy()
            value = float(value)  # Requires np float and not jax.numpy float
            loss_hist.append(value)

            rho_final = rho_final.detach().cpu().numpy()

        config.cur_it += 1
        print(f"value: {value}")
        print(f"grad: {np.mean(grad)}")
        print(f"iteration: {config.cur_it}")
        print(f"time: {time.time() - start}")

        # if np.abs(prev_val - value) <= 1e-4:
        #     break
        # else:
        #     prev_val = value

        updates, opt_state = optimizer.update(-grad, opt_state, rho_0)

        rho[:] = optax.apply_updates(rho_0, updates)
        np.clip(rho, 0.0, 1.0, out=rho)

        if value > best_val:
            rho_opt = rho.copy()
            best_val = value
        if eval:
            plotter(rho_init, rho_final, projection, config.cur_it)



    return rho_opt, loss_hist, em_hist, grad_hist

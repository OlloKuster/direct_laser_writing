import jax
import numpy as np
import torch
import nlopt
import optax
import time
import matplotlib.pyplot as plt

from filtering.dose_model.config_print import ConfigPrint
from optimizer import config


def optimizer_nlopt(rho, objective, filter, projection, init_projection, plotter, mode, eval=False):
    """
    The optimiser function
    :return:
    """
    loss_hist = []
    em_hist = []
    grad_hist = []

    def select_f(mode):
        if mode == "torch_jax":
            return f_torch_jax
        if mode == "jax":
            return f_jax

    class fom_em_torch_f(torch.autograd.Function):
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

    def f_torch_jax(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        x = np.array(init_projection(x))
        rho_0 = torch.tensor(x, device='cuda', requires_grad=True)
        rho_final = filter(rho_0)
        fom = fom_em_torch_f.apply(rho_final)
        fom.backward(retain_graph=True)
        grad_torch = rho_0.grad
        value = fom.detach().cpu().numpy()
        grad = grad_torch.detach().cpu().numpy()
        value = float(value)  # Requires np float and not jax.numpy float
        grad_hist.append(np.sum(grad))
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            plotter(x, rho_final.detach().cpu().numpy(), projection, config.cur_it)
        return value

    def f_jax(x, g):
        start = time.time()
        rho_0 = np.reshape(x, rho.shape)
        rho_final = filter(rho_0)
        value_obj, grad = jax.value_and_grad(objective, has_aux=True)(rho_final)
        value = float(value_obj[0])  # Requires np float and not jax.numpy float
        value_em = float(value_obj[1])
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        em_hist.append(value_em)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            plotter(rho_0, rho_final, projection, config.ind)
            if config.cur_it % config.MAXEVAL == 0:
                config.ind += 1
        return value

    f = select_f(mode)

    opt = nlopt.opt(config.OPTIMISER, rho.size)
    # opt.set_param('tolg', 1e-12)
    opt.set_min_objective(f)
    opt.set_maxeval(config.MAXEVAL)
    # opt.set_ftol_abs(config.FTOL_ABS)
    # opt.set_ftol_rel(config.FTOL_REL)
    # opt.set_stopval(1e-4)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(rho.ravel())
    rho_opt = rho_opt.reshape(rho.shape)

    return rho_opt, loss_hist, em_hist, grad_hist


def optimizer_optax(rho, objective, filter, projection, init_projection, plotter, mode, eval=False):
    rho = np.array(rho)

    round = (config.cur_it + 1) / config.MAXEVAL
    # print(round)
    # if round >= 2:
    #     lr = 1e-2
    # else:
    #     lr = config.lr
    optimizer = optax.adam(learning_rate=config.lr)
    opt_state = optimizer.init(rho)

    loss_hist = []
    em_hist = []
    grad_hist = []

    rho_opt = rho
    best_val = 100

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

    for i in range(config.MAXEVAL):
        start = time.time()
        rho_init = np.array(init_projection(rho))
        if mode == "jax":
            rho_final = filter(rho_init)
            value_obj, grad = jax.value_and_grad(objective, has_aux=True)(rho_final)
            value = float(value_obj[0])  # Requires np float and not jax.numpy float
            value_em = float(value_obj[1])
            grad = np.array(grad)
            loss_hist.append(value)
            em_hist.append(value_em)

        if mode == "torch_jax":
            rho = torch.tensor(rho_init, device='cuda', requires_grad=True)
            rho_final = filter(rho)
            fom = FomEmTorchF.apply(rho_final)
            fom.backward(retain_graph=True)
            grad_torch = rho.grad
            rho = rho.detach().cpu().numpy()
            value = fom.detach().cpu().numpy()
            grad = grad_torch.detach().cpu().numpy()
            value = float(value)  # Requires np float and not jax.numpy float
            loss_hist.append(value)

        config.cur_it += 1
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        print(f"time: {time.time() - start}")

        if value < best_val:
            rho_opt = rho
        if eval:
            plotter(rho_init, rho_final.detach().cpu().numpy(), projection, config.cur_it)

        updates, opt_state = optimizer.update(grad, opt_state, rho)
        rho[:] = optax.apply_updates(rho, updates)

        np.clip(rho, 0.0, 1.0, out=rho)

    return rho_opt, loss_hist, em_hist, grad_hist

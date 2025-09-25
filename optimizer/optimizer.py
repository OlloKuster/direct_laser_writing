import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import autograd as ag
import nlopt
import time

from optimizer import config


def optimiser_jaxwell(rho, dose_sim, objective, eval=False):
    """
    The optimiser function
    :return:
    """
    loss_hist = []

    class fom_em_torch_f(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            grad_em_sim_f = jax.value_and_grad(objective)
            value_em_sim, grad_em_sim = grad_em_sim_f(x.detach().cpu().numpy().astype(np.float64))
            ctx.save_for_backward(torch.tensor(np.array(grad_em_sim), device='cuda', requires_grad=True))
            return torch.tensor(np.array(value_em_sim), device='cuda', requires_grad=True)

        @staticmethod
        def backward(ctx, grad):
            grad_em_sim, = ctx.saved_tensors
            return grad_em_sim

    def f(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        rho_0 = torch.tensor(x, device='cuda', requires_grad=True)
        rho_final = dose_sim(rho_0)
        fom = fom_em_torch_f.apply(rho_final)
        fom.backward(retain_graph=True)
        grad_torch = rho_0.grad
        value = fom.detach().cpu().numpy()
        grad = grad_torch.detach().cpu().numpy()
        value = float(value)  # Requires np float and not jax.numpy float
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        return value

    opt = nlopt.opt(config.OPTIMISER, rho.size)
    opt.set_max_objective(f)
    opt.set_maxeval(config.MAXEVAL)
    # opt.set_ftol_abs(config.FTOL_ABS)
    # opt.set_ftol_rel(config.FTOL_REL)
    # opt.set_stopval(1e-4)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(rho.ravel())
    rho_opt = rho_opt.reshape(rho.shape)

    return rho_opt, loss_hist

